from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate, StatePreparation
from qiskit.quantum_info import Operator, Statevector


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "gatsby_5.16.1_dependencies_labeled.json"
DEFAULT_INFECTED = "es-errors@1.3.0"
DEFAULT_TOP_K = 50


@dataclass(frozen=True)
class RankingResult:
    method: str
    ranking: list[str]
    scores: dict[str, float]
    runtime_seconds: float = 0.0


def default_data_path() -> Path:
    choices = [
        PROJECT_DIR / "data" / DATA_FILENAME,
        PROJECT_DIR / DATA_FILENAME,
        PROJECT_DIR / "upload" / DATA_FILENAME,
    ]
    return next((path for path in choices if path.is_file()), choices[0])


def package_label(raw_node: dict) -> str:
    key = raw_node["versionKey"]
    return f"{key['name']}@{key['version']}"


def stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)


def load_dataset(
    path: Path,
) -> tuple[nx.DiGraph, dict[str, int], dict, dict]:
    """
    그래프와 정답 라벨을 서로 분리해 읽는다.

    반환되는 graph에는 infection_label을 넣지 않는다. 따라서 순위 알고리즘은
    그래프 객체를 받아도 정답 라벨에 접근할 수 없다. labels는 모든 순위 계산이
    끝난 뒤 evaluate_ranking에서만 사용한다.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    if not raw_nodes:
        raise ValueError(f"노드가 없는 JSON입니다: {path}")

    raw_labels = [package_label(node) for node in raw_nodes]
    dependency_graph = nx.DiGraph()
    labels: dict[str, int] = {}
    duplicate_count = 0

    for node_name, raw_node in zip(raw_labels, raw_nodes):
        infection_label = raw_node.get("infection_label")
        if infection_label not in (0, 1):
            raise ValueError(
                f"{node_name}의 infection_label이 0 또는 1이 아닙니다: "
                f"{infection_label!r}"
            )
        if node_name in labels:
            duplicate_count += 1
            if labels[node_name] != infection_label:
                raise ValueError(
                    f"중복 package@version의 감염 라벨이 충돌합니다: {node_name}"
                )
        labels[node_name] = int(infection_label)

        key = raw_node["versionKey"]
        dependency_graph.add_node(
            node_name,
            system=key.get("system", "NPM"),
            package_name=key["name"],
            version=key["version"],
        )

    removed_self_loops = 0
    for edge in raw_edges:
        dependent = raw_labels[edge["fromNode"]]
        dependency = raw_labels[edge["toNode"]]
        if dependent == dependency:
            removed_self_loops += 1
            continue
        dependency_graph.add_edge(dependent, dependency)

    stats = {
        "raw_node_count": len(raw_nodes),
        "unique_node_count": dependency_graph.number_of_nodes(),
        "duplicate_package_version_rows": duplicate_count,
        "raw_edge_count": len(raw_edges),
        "unique_edge_count": dependency_graph.number_of_edges(),
        "removed_self_loops": removed_self_loops,
    }
    metadata = data.get("infection_label_metadata", {})
    return dependency_graph, labels, metadata, stats


def make_propagation_graph(dependency_graph: nx.DiGraph) -> nx.DiGraph:
    """A가 B에 의존하는 A->B를 악성 영향 전파 방향 B->A로 뒤집는다."""
    return dependency_graph.reverse(copy=True)


def make_candidate_subgraph(
    graph: nx.DiGraph,
    infected: str,
    candidate_limit: int,
    top_k: int,
) -> tuple[nx.DiGraph, set[str], dict[str, int], int]:
    if infected not in graph:
        raise ValueError(f"초기 감염원이 그래프에 없습니다: {infected}")

    all_distances = dict(nx.single_source_shortest_path_length(graph, infected))
    ordered = sorted(
        set(all_distances) - {infected},
        key=lambda node: (
            all_distances[node],
            -graph.out_degree(node),
            node,
        ),
    )
    reachable_count = len(ordered)
    if candidate_limit > 0:
        ordered = ordered[:candidate_limit]
    if len(ordered) < top_k:
        raise ValueError(
            f"Top-{top_k}를 뽑아야 하지만 전파 후보가 {len(ordered)}개뿐입니다."
        )

    candidates = set(ordered)
    subgraph = graph.subgraph({infected, *candidates}).copy()
    distances = {
        node: distance
        for node, distance in all_distances.items()
        if node == infected or node in candidates
    }
    return subgraph, candidates, distances, reachable_count


def assign_transmission_probabilities(
    graph: nx.DiGraph,
    base_probability: float,
    heterogeneity: float,
    seed: int,
) -> None:
    """감염 라벨을 보지 않고 간선 이름과 고정 seed만으로 확률을 만든다."""
    for source, target in graph.edges:
        centered = (
            2.0 * stable_unit_interval(f"{seed}|{source}|{target}") - 1.0
        )
        probability = base_probability * (1.0 + heterogeneity * centered)
        probability = min(0.999999, max(0.000001, probability))
        graph[source][target]["transmission_probability"] = probability
        graph[source][target]["transmission_distance"] = -math.log(probability)


def rank_bfs(
    graph: nx.DiGraph,
    candidates: set[str],
    distances: dict[str, int],
) -> RankingResult:
    ranking = sorted(
        candidates,
        key=lambda node: (distances[node], -graph.out_degree(node), node),
    )
    scores = {node: 1.0 / (1.0 + distances[node]) for node in candidates}
    return RankingResult("bfs", ranking, scores)


def rank_personalized_pagerank(
    graph: nx.DiGraph,
    infected: str,
    candidates: set[str],
    distances: dict[str, int],
    alpha: float,
) -> RankingResult:
    personalization = {node: 0.0 for node in graph}
    personalization[infected] = 1.0
    all_scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        dangling=personalization,
        weight="transmission_probability",
        max_iter=2_000,
        tol=1.0e-12,
    )
    scores = {node: float(all_scores[node]) for node in candidates}
    ranking = sorted(
        candidates,
        key=lambda node: (-scores[node], distances[node], node),
    )
    return RankingResult("personalized_pagerank", ranking, scores)


def rank_betweenness(
    graph: nx.DiGraph,
    candidates: set[str],
    distances: dict[str, int],
) -> RankingResult:
    all_scores = nx.betweenness_centrality(
        graph,
        normalized=True,
        weight="transmission_distance",
    )
    scores = {node: float(all_scores[node]) for node in candidates}
    ranking = sorted(
        candidates,
        key=lambda node: (-scores[node], distances[node], node),
    )
    return RankingResult("betweenness_centrality", ranking, scores)


def simulate_independent_cascade(
    graph: nx.DiGraph,
    infected: str,
    random_seed: int,
) -> set[str]:
    rng = random.Random(random_seed)
    infected_nodes = {infected}
    frontier = [infected]
    while frontier:
        next_frontier: set[str] = set()
        for source in sorted(frontier):
            for target in sorted(graph.successors(source)):
                if target in infected_nodes:
                    continue
                probability = graph[source][target]["transmission_probability"]
                if rng.random() < probability:
                    infected_nodes.add(target)
                    next_frontier.add(target)
        frontier = sorted(next_frontier)
    return infected_nodes


def rank_monte_carlo(
    graph: nx.DiGraph,
    infected: str,
    candidates: set[str],
    distances: dict[str, int],
    simulations: int,
    seed: int,
) -> RankingResult:
    counts = {node: 0 for node in candidates}
    for simulation_id in range(simulations):
        infected_nodes = simulate_independent_cascade(
            graph,
            infected,
            seed + simulation_id * 1_000_003,
        )
        for node in infected_nodes & candidates:
            counts[node] += 1
    scores = {node: counts[node] / simulations for node in candidates}
    ranking = sorted(
        candidates,
        key=lambda node: (-scores[node], distances[node], node),
    )
    return RankingResult("monte_carlo_infection_risk", ranking, scores)


def build_chiral_hamiltonian(
    graph: nx.DiGraph,
    ordered_nodes: Sequence[str],
    phase: float,
) -> np.ndarray:
    index = {node: position for position, node in enumerate(ordered_nodes)}
    adjacency = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=float)
    for source, target, attributes in graph.edges(data=True):
        adjacency[index[target], index[source]] += math.sqrt(
            attributes["transmission_probability"]
        )
    symmetric = 0.5 * (adjacency + adjacency.T)
    antisymmetric = 0.5 * (adjacency - adjacency.T)
    hamiltonian = (
        math.cos(phase) * symmetric
        + 1j * math.sin(phase) * antisymmetric
    )
    if not np.allclose(hamiltonian, hamiltonian.conjugate().T):
        raise RuntimeError("CTQW Hamiltonian이 Hermitian이 아닙니다.")
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    radius = float(np.max(np.abs(eigenvalues)))
    return hamiltonian / radius if radius > 0.0 else hamiltonian


def qiskit_register_size(logical_dimension: int) -> tuple[int, int]:
    qubits = max(1, (logical_dimension - 1).bit_length())
    return qubits, 1 << qubits


def qiskit_time_average_probabilities(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time_max: float,
    time_steps: int,
) -> np.ndarray:
    logical_dimension = len(initial_state)
    qubits, qiskit_dimension = qiskit_register_size(logical_dimension)

    padded_state = np.zeros(qiskit_dimension, dtype=np.complex128)
    padded_state[:logical_dimension] = initial_state
    preparation = QuantumCircuit(qubits)
    preparation.append(StatePreparation(padded_state), preparation.qubits)
    statevector = Statevector.from_instruction(preparation)

    padded_hamiltonian = np.zeros(
        (qiskit_dimension, qiskit_dimension), dtype=np.complex128
    )
    padded_hamiltonian[:logical_dimension, :logical_dimension] = hamiltonian
    evolution = QuantumCircuit(qubits)
    evolution.append(
        HamiltonianGate(padded_hamiltonian, time_max / time_steps),
        evolution.qubits,
    )
    step_operator = Operator(evolution)

    accumulated = np.zeros(logical_dimension, dtype=float)
    for _ in range(time_steps):
        statevector = statevector.evolve(step_operator)
        probabilities = np.asarray(
            statevector.probabilities()[:logical_dimension], dtype=float
        )
        accumulated += probabilities / (float(np.sum(probabilities)) or 1.0)
    return accumulated / time_steps


def rank_directional_ctqw(
    graph: nx.DiGraph,
    infected: str,
    candidates: set[str],
    distances: dict[str, int],
    time_max: float,
    time_steps: int,
    phase: float,
) -> RankingResult:
    """
    감염 라벨을 모르는 방향성 CTQW 위험도 휴리스틱이다.

    infection_label projector/oracle은 사용하지 않는다. 즉 라벨은 Hamiltonian,
    초기상태, 진화시간, 위상, 점수 계산 어디에도 들어가지 않는다.
    """
    ordered_nodes = [infected, *sorted(candidates)]
    index = {node: position for position, node in enumerate(ordered_nodes)}
    hamiltonian = build_chiral_hamiltonian(graph, ordered_nodes, phase)
    initial_state = np.zeros(len(ordered_nodes), dtype=np.complex128)
    initial_state[index[infected]] = 1.0
    probabilities = qiskit_time_average_probabilities(
        hamiltonian,
        initial_state,
        time_max,
        time_steps,
    )
    scores = {node: float(probabilities[index[node]]) for node in candidates}
    ranking = sorted(
        candidates,
        key=lambda node: (-scores[node], distances[node], node),
    )
    return RankingResult("phase_ctqw_risk_heuristic", ranking, scores)


def measure(
    factory: Callable[[], RankingResult],
    repeats: int,
) -> RankingResult:
    durations: list[float] = []
    result: RankingResult | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        current = factory()
        durations.append(time.perf_counter() - start)
        if result is None:
            result = current
    if result is None:
        raise RuntimeError("순위 계산에 실패했습니다.")
    return RankingResult(
        result.method,
        result.ranking,
        result.scores,
        statistics.median(durations),
    )


def evaluate_ranking(
    result: RankingResult,
    infection_labels: dict[str, int],
    candidates: set[str],
    top_k: int,
) -> dict:
    """이 함수에서만 infection_label을 읽는다."""
    predicted = set(result.ranking[:top_k])
    actual = {node for node in candidates if infection_labels[node] == 1}
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    precision = tp / len(predicted)
    recall = tp / len(actual) if actual else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "method": result.method,
        "top_k": top_k,
        "true_infected_count": len(actual),
        "predicted_infected_count": len(predicted),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "runtime_seconds": result.runtime_seconds,
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"쓸 행이 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    data_path: Path,
    infected: str,
    candidates: set[str],
    distances: dict[str, int],
    infection_labels: dict[str, int],
    label_metadata: dict,
    graph_stats: dict,
    rankings: Sequence[RankingResult],
    comparison_rows: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2 input: intentionally excludes the evaluation-only infection_label.
    stage2_rows: list[dict] = []
    for result in rankings:
        for rank, node in enumerate(result.ranking[: args.top_k], start=1):
            stage2_rows.append(
                {
                    "method": result.method,
                    "rank": rank,
                    "node": node,
                    "risk_score": result.scores[node],
                    "predicted_infected": 1,
                    "propagation_distance": distances[node],
                }
            )
    write_csv(output_dir / "rankings.csv", stage2_rows)

    # Evaluation-only audit output, kept separate from the Stage 2 input.
    audit_rows: list[dict] = []
    for result in rankings:
        rank_by_node = {
            node: rank for rank, node in enumerate(result.ranking, start=1)
        }
        for node in sorted(candidates):
            rank = rank_by_node[node]
            audit_rows.append(
                {
                    "method": result.method,
                    "node": node,
                    "rank": rank,
                    "risk_score": result.scores[node],
                    "predicted_infected": int(rank <= args.top_k),
                    "infection_label": infection_labels[node],
                }
            )
    write_csv(output_dir / "stage1_predictions_with_labels.csv", audit_rows)
    write_csv(output_dir / "method_comparison.csv", comparison_rows)
    write_csv(
        output_dir / "infection_ground_truth.csv",
        [
            {"node": node, "infection_label": infection_labels[node]}
            for node in sorted(candidates)
        ],
    )

    qubits, dimension = qiskit_register_size(len(candidates) + 1)
    summary = {
        "experiment": "infected-candidate detection",
        "data": str(data_path.resolve()),
        "known_infected_source": infected,
        "candidate_count": len(candidates),
        "selected_candidate_count_per_method": args.top_k,
        "graph_stats": graph_stats,
        "label_metadata": label_metadata,
        "configuration": {
            "base_probability": args.infection_probability,
            "probability_heterogeneity": args.probability_heterogeneity,
            "probability_seed": args.probability_seed,
            "pagerank_alpha": args.pagerank_alpha,
            "monte_carlo_simulations": args.monte_carlo_simulations,
            "ctqw_time_max": args.ctqw_time_max,
            "ctqw_time_steps": args.ctqw_time_steps,
            "ctqw_phase": args.ctqw_phase,
        },
        "ctqw": {
            "framework": f"Qiskit {qiskit.__version__}",
            "simulation": "exact Statevector on a classical computer",
            "logical_states": len(candidates) + 1,
            "qubits": qubits,
            "padded_dimension": dimension,
            "infection_oracle_used": False,
            "real_quantum_hardware_used": False,
        },
        "leakage_guard": {
            "allowed_model_inputs": [
                "dependency edge direction",
                "known initial infected source",
                "deterministic synthetic edge probability",
                "graph-derived distance and centrality",
            ],
            "forbidden_model_input": "infection_label",
            "ranking_file_contains_infection_label": False,
            "label_usage": "evaluation only after every ranking is frozen",
            "ctqw_parameter_selection": "fixed before label evaluation",
            "oracle_search_disabled": True,
        },
        "method_comparison": list(comparison_rows),
        "ranking_rule": f"F1@{args.top_k} descending; runtime is not used for ranking",
        "claim_limit": (
            "The embedded labels are a synthetic CTQW-favorable toy benchmark. "
            "These results cannot establish real-world accuracy or quantum advantage."
        ),
    }
    with (output_dir / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "감염 라벨은 평가에만 사용하고, 그래프 구조만으로 각 방법의 "
            "그래프 구조만으로 감염 후보를 순위화하고 Top-K 성능을 비교합니다."
        )
    )
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--infected", default=DEFAULT_INFECTED)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"각 방법에서 Stage 2로 전달할 상위 후보 수 (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="0이면 감염원에서 도달 가능한 후보 전체를 사용합니다.",
    )
    parser.add_argument("--infection-probability", type=float, default=0.55)
    parser.add_argument("--probability-heterogeneity", type=float, default=0.25)
    parser.add_argument("--probability-seed", type=int, default=7_301)
    parser.add_argument("--pagerank-alpha", type=float, default=0.85)
    parser.add_argument("--monte-carlo-simulations", type=int, default=2_000)
    parser.add_argument("--monte-carlo-seed", type=int, default=81_071)
    parser.add_argument("--ctqw-time-max", type=float, default=20.0)
    parser.add_argument("--ctqw-time-steps", type=int, default=200)
    parser.add_argument("--ctqw-phase", type=float, default=math.pi / 4.0)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise ValueError(f"라벨이 추가된 데이터가 없습니다: {args.data}")
    if args.top_k <= 0:
        raise ValueError("top-k는 양수여야 합니다.")
    if 0 < args.candidate_limit < args.top_k:
        raise ValueError(f"candidate-limit은 0 또는 {args.top_k} 이상이어야 합니다.")
    if not 0.0 < args.infection_probability < 1.0:
        raise ValueError("infection-probability은 0과 1 사이여야 합니다.")
    if not 0.0 <= args.probability_heterogeneity < 1.0:
        raise ValueError("probability-heterogeneity는 0 이상 1 미만이어야 합니다.")
    if not 0.0 < args.pagerank_alpha < 1.0:
        raise ValueError("pagerank-alpha는 0과 1 사이여야 합니다.")
    if min(
        args.monte_carlo_simulations,
        args.ctqw_time_steps,
        args.runtime_repeats,
    ) <= 0:
        raise ValueError("반복 횟수는 양수여야 합니다.")
    if args.ctqw_time_max <= 0.0:
        raise ValueError("ctqw-time-max는 양수여야 합니다.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dependency_graph, labels, label_metadata, graph_stats = load_dataset(args.data)
    propagation_graph = make_propagation_graph(dependency_graph)
    subgraph, candidates, distances, reachable_count = make_candidate_subgraph(
        propagation_graph,
        args.infected,
        args.candidate_limit,
        args.top_k,
    )
    if labels.get(args.infected) != 1:
        raise ValueError("알려진 초기 감염원의 infection_label은 1이어야 합니다.")

    assign_transmission_probabilities(
        subgraph,
        args.infection_probability,
        args.probability_heterogeneity,
        args.probability_seed,
    )

    factories: list[Callable[[], RankingResult]] = [
        lambda: rank_bfs(subgraph, candidates, distances),
        lambda: rank_personalized_pagerank(
            subgraph, args.infected, candidates, distances, args.pagerank_alpha
        ),
        lambda: rank_betweenness(subgraph, candidates, distances),
        lambda: rank_monte_carlo(
            subgraph,
            args.infected,
            candidates,
            distances,
            args.monte_carlo_simulations,
            args.monte_carlo_seed,
        ),
        lambda: rank_directional_ctqw(
            subgraph,
            args.infected,
            candidates,
            distances,
            args.ctqw_time_max,
            args.ctqw_time_steps,
            args.ctqw_phase,
        ),
    ]
    rankings = [measure(factory, args.runtime_repeats) for factory in factories]

    comparison_rows = [
        evaluate_ranking(result, labels, candidates, args.top_k)
        for result in rankings
    ]
    comparison_rows.sort(key=lambda row: (-row["f1"], row["method"]))
    comparison_rows = [
        {"performance_rank": rank, **row}
        for rank, row in enumerate(comparison_rows, start=1)
    ]

    write_outputs(
        args.output_dir,
        args.data,
        args.infected,
        candidates,
        distances,
        labels,
        label_metadata,
        {**graph_stats, "reachable_candidate_count": reachable_count},
        rankings,
        comparison_rows,
        args,
    )

    print(f"Known infected source: {args.infected}")
    print(f"Reachable candidates: {len(candidates)}")
    print(f"Each method forwards exactly {args.top_k} ranked candidates")
    print("infection_label input leakage: blocked (evaluation only)\n")
    print(
        f"{'rank':>4}  {'method':29} {('F1@' + str(args.top_k)):>8} "
        f"{'precision':>10} {'recall':>8} {'time(s)':>10}"
    )
    for row in comparison_rows:
        print(
            f"{row['performance_rank']:4d}  {row['method'][:29]:29} "
            f"{row['f1']:8.4f} {row['precision']:10.4f} "
            f"{row['recall']:8.4f} {row['runtime_seconds']:10.6f}"
        )
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
