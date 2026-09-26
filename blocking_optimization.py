from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from scipy.optimize import minimize


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data" / "gatsby_5.16.1_dependencies_labeled.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "stage2_outputs"

DEFAULT_CANDIDATE_METHOD = "phase_ctqw_risk_heuristic"
DEFAULT_STAGE1_CANDIDATE_COUNT = 32
DEFAULT_QUBO_CANDIDATE_COUNT = 15
DEFAULT_BLOCK_COUNT = 5
DEFAULT_AVAILABILITY_COST_WEIGHT = 0.5
DEFAULT_SHOTS = 10_000


@dataclass(frozen=True)
class QuboModel:
    nodes: list[str]
    linear: np.ndarray
    quadratic: dict[tuple[int, int], float]
    constant: float
    single_block_reductions: np.ndarray
    pair_interactions: dict[tuple[int, int], float]
    prevented_by_single_block: dict[str, set[str]]
    baseline_infected: frozenset[str]
    infection_sources: tuple[str, ...]


@dataclass(frozen=True)
class MethodResult:
    method: str
    selected: tuple[int, ...]
    objective: float
    runtime_seconds: float
    details: dict


@dataclass(frozen=True)
class SpreadOracle:
    selected: tuple[int, ...]
    remaining_infected_count: int
    infection_reduction_count: int
    runtime_seconds: float
    combinations_checked: int
    best_qubo_energy: float
    best_qubo_selected: tuple[int, ...]
    best_qubo_reduction: int


def package_label(raw_node: dict) -> str:
    key = raw_node["versionKey"]
    return f"{key['name']}@{key['version']}"


def default_data_path() -> Path:
    filename = "gatsby_5.16.1_dependencies_labeled.json"
    choices = [
        PROJECT_DIR / "data" / filename,
        PROJECT_DIR / filename,
        PROJECT_DIR / "upload" / filename,
        PROJECT_DIR.parent / filename,
    ]
    return next((path for path in choices if path.is_file()), DEFAULT_DATA)


def default_rankings_path() -> Path:
    choices = [
        PROJECT_DIR / "outputs" / "rankings.csv",
        PROJECT_DIR / "phase1_ctqw_candidates.csv",
    ]
    return next((path for path in choices if path.is_file()), choices[0])


def load_propagation_graph(path: Path) -> nx.DiGraph:
    """JSON에서 의존성 전파 그래프만 읽는다.

    2단계는 감염 패키지를 다시 분류하는 단계가 아니다. 따라서 JSON에
    ``infection_label``이 있더라도 이 함수는 해당 필드를 읽지도, 검증하지도
    않는다. 알려진 최초 감염원에서 그래프를 따라 확산되는 결과만 사용한다.
    """
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    raw_nodes = data.get("nodes", [])
    labels = [package_label(node) for node in raw_nodes]
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(labels)
    for edge in data.get("edges", []):
        source = labels[edge["fromNode"]]
        target = labels[edge["toNode"]]
        if source != target:
            dependency_graph.add_edge(source, target)
    # deps.dev 의존성 간선과 실제 감염 전파 방향이 반대이므로 뒤집는다.
    return dependency_graph.reverse(copy=True)


def load_stage1_candidates(
    path: Path,
    method: str,
    stage1_candidate_count: int,
    qubo_candidate_count: int,
) -> tuple[list[str], list[str]]:
    """
    Stage 1 랭킹에서 지정된 수의 후보를 읽고 상위 N개 노드만 반환한다.

    2단계 목적은 후보의 감염 여부 재분류가 아니라 차단 조합 최적화다.
    따라서 rankings.csv의 risk_score, score, infection_label 등은 사용하지
    않고 오직 1단계가 확정한 순위와 노드명만 후보 구성에 사용한다.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("method") == method]

    rows.sort(key=lambda row: int(row["rank"]))
    stage1_pool: list[str] = []
    for row in rows:
        node = row["node"]
        if node not in stage1_pool:
            stage1_pool.append(node)
        if len(stage1_pool) == stage1_candidate_count:
            break

    if len(stage1_pool) != stage1_candidate_count:
        raise ValueError(
            f"'{method}'의 1단계 후보가 {stage1_candidate_count}개 필요하지만 "
            f"랭킹 파일에서 {len(stage1_pool)}개만 찾았습니다."
        )
    shortlist = stage1_pool[:qubo_candidate_count]
    return stage1_pool, shortlist


def simulate_infections(
    graph: nx.DiGraph,
    infection_sources: Sequence[str],
    blocked_nodes: Sequence[str] | set[str] = (),
) -> set[str]:
    """차단 노드를 통과하지 않는 결정론적 BFS 확산 결과를 반환한다."""
    blocked = set(blocked_nodes)
    queue = [source for source in infection_sources if source not in blocked]
    infected = set(queue)
    position = 0
    while position < len(queue):
        node = queue[position]
        position += 1
        for neighbor in graph.successors(node):
            if neighbor not in blocked and neighbor not in infected:
                infected.add(neighbor)
                queue.append(neighbor)
    return infected


def build_qubo(
    graph: nx.DiGraph,
    candidates: Sequence[str],
    infection_sources: Sequence[str],
    block_count: int,
    availability_cost_weight: float,
) -> QuboModel:
    """감염 감소량을 최대화하는 2차 근사 QUBO를 만든다.

    후보 ``i`` 하나를 차단했을 때 실제 BFS 감염 수가 얼마나 감소하는지를
    ``r_i``로 두고, 후보 ``i, j``를 함께 차단했을 때 새로 생기는 중복/시너지
    효과를 ``d_ij = r_ij - r_i - r_j``로 계산한다.

    maximize  sum(r_i*x_i) + sum(d_ij*x_i*x_j)
    subject to sum(x_i) = K

    QAOA는 최소화 해밀토니안을 사용하므로 방어 효과 항의 부호를 뒤집고,
    dependency in-degree 기반 availability cost를 함께 반영한다. 정확히 K개 선택은
    fixed-Hamming-weight 초기 상태와 XY mixer가 회로 구조상 보존한다.
    고차 상호작용은 QUBO가 표현할 수 없어 최종 평가는 전체 BFS 재시뮬레이션으로 수행한다.
    """
    unknown_candidates = sorted(set(candidates) - set(graph))
    if unknown_candidates:
        raise ValueError(
            f"그래프에 없는 후보 노드가 있습니다: {unknown_candidates[:5]}"
        )
    unknown_sources = sorted(set(infection_sources) - set(graph))
    if unknown_sources:
        raise ValueError(
            f"그래프에 없는 최초 감염원이 있습니다: {unknown_sources}"
        )

    sources = tuple(dict.fromkeys(infection_sources))
    baseline_infected = simulate_infections(graph, sources)
    baseline_count = len(baseline_infected)
    if baseline_count == 0:
        raise ValueError("최초 감염원에서 도달 가능한 감염 패키지가 없습니다.")

    prevented_by_single_block: dict[str, set[str]] = {}
    single_reductions: list[float] = []
    for node in candidates:
        remaining = simulate_infections(graph, sources, {node})
        prevented = baseline_infected - remaining
        prevented_by_single_block[node] = prevented
        single_reductions.append(float(len(prevented)))

    pair_interactions: dict[tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(len(candidates)), 2):
        remaining = simulate_infections(
            graph,
            sources,
            {candidates[i], candidates[j]},
        )
        pair_reduction = float(baseline_count - len(remaining))
        pair_interactions[i, j] = (
            pair_reduction - single_reductions[i] - single_reductions[j]
        )

    # baseline_count로 나누면 회로 회전각이 그래프 크기에 비례해 폭증하지 않는다.
    scale = float(baseline_count)
    normalized_single = np.asarray(single_reductions, dtype=float) / scale
    normalized_interactions = {
        pair: value / scale for pair, value in pair_interactions.items()
    }

    # 각 노드의 중요도 점수를 계산하거나 불러옵니다.
    # 예시로 in-degree(자신을 의존하는 패키지 수)를 중요도(비용)로 사용.
    importance_scores = [graph.in_degree(node) for node in candidates]
    
    # 중요도 점수를 최대 1.0이 되도록 정규화
    max_importance = max(importance_scores) if max(importance_scores) > 0 else 1.0
    normalized_costs = np.array(importance_scores, dtype=float) / max_importance

    # 방어 효과와 정상 서비스 영향(availability cost) 사이의 trade-off.
    base_linear = -normalized_single + (
        availability_cost_weight * normalized_costs
    )

    base_quadratic = {
        pair: -interaction for pair, interaction in normalized_interactions.items()
    }

    # P(sum x_i - K)^2 = P[(1-2K)sum x_i + 2sum_{i<j}x_ix_j + K^2]
    # linear = base_linear + cardinality_penalty * (1.0 - 2.0 * block_count)
    # quadratic = {
    #     pair: base_quadratic[pair] + 2.0 * cardinality_penalty
    #     for pair in pair_interactions
    # }
    linear = base_linear.copy()
    quadratic = base_quadratic.copy()
    constant = 0.0
    return QuboModel(
        nodes=list(candidates),
        linear=linear,
        quadratic=quadratic,
        constant=constant,
        single_block_reductions=np.asarray(single_reductions, dtype=float),
        pair_interactions=pair_interactions,
        prevented_by_single_block=prevented_by_single_block,
        baseline_infected=frozenset(baseline_infected),
        infection_sources=sources,
    )


def qubo_energy(model: QuboModel, selected: Sequence[int]) -> float:
    chosen = set(selected)
    energy = model.constant + sum(model.linear[i] for i in chosen)
    energy += sum(
        coefficient
        for (i, j), coefficient in model.quadratic.items()
        if i in chosen and j in chosen
    )
    return float(energy)


def spread_counts_for_indices(
    graph: nx.DiGraph,
    model: QuboModel,
    selected: Sequence[int],
) -> tuple[int, int]:
    blocked_nodes = {model.nodes[index] for index in selected}
    remaining = simulate_infections(
        graph,
        model.infection_sources,
        blocked_nodes,
    )
    remaining_count = len(remaining)
    reduction_count = len(model.baseline_infected) - remaining_count
    return remaining_count, reduction_count


def exact_spread_oracle(
    graph: nx.DiGraph,
    model: QuboModel,
    block_count: int,
    max_combinations: int,
) -> SpreadOracle | None:
    """작은 toy model에서 실제 BFS 감염 감소량의 정확한 최댓값을 구한다."""
    total = math.comb(len(model.nodes), block_count)
    if total > max_combinations:
        return None

    start = time.perf_counter()
    best: tuple[int, ...] | None = None
    best_remaining = len(model.baseline_infected)
    best_reduction = -1
    best_qubo_energy = math.inf
    qb_best_selected = None
    qb_best_energy = math.inf
    qb_best_reduction = -1
    checked = 0
    for selected in itertools.combinations(range(len(model.nodes)), block_count):
        checked += 1
        remaining, reduction = spread_counts_for_indices(graph, model, selected)
        energy = qubo_energy(model, selected)
        if (
            reduction > best_reduction
            or (
                reduction == best_reduction
                and (
                    energy < best_qubo_energy - 1.0e-12
                    or (
                        math.isclose(energy, best_qubo_energy, abs_tol=1.0e-12)
                        and (best is None or selected < best)
                    )
                )
            )
        ):
            best = tuple(selected)
            best_remaining = remaining
            best_reduction = reduction
            best_qubo_energy = energy

            if energy < qb_best_energy - 1e-12:
                qb_best_energy = energy
                qb_best_selected = tuple(selected)
                qb_best_reduction = reduction

    if best is None:
        raise RuntimeError("정확 감염 감소량 탐색에서 해를 찾지 못했습니다.")
    return SpreadOracle(
        selected=best,
        remaining_infected_count=best_remaining,
        infection_reduction_count=best_reduction,
        runtime_seconds=time.perf_counter() - start,
        combinations_checked=checked,
        best_qubo_energy=qb_best_energy,          
        best_qubo_selected=qb_best_selected,      
        best_qubo_reduction=qb_best_reduction,
    )


def greedy(
    graph: nx.DiGraph,
    model: QuboModel,
    block_count: int,
) -> MethodResult:
    """매 단계 실제 BFS 감염 수를 가장 많이 줄이는 후보를 추가한다."""
    start = time.perf_counter()
    selected: list[int] = []
    while len(selected) < block_count:
        remaining = set(range(len(model.nodes))) - set(selected)
        choice = min(
            remaining,
            key=lambda node: (
                qubo_energy(model, [*selected, node]),
                # spread_counts_for_indices(graph, model, [*selected, node])[0],
                model.nodes[node],
            ),
        )
        selected.append(choice)
    return MethodResult(
        "Greedy",
        tuple(sorted(selected)),
        qubo_energy(model, selected),
        time.perf_counter() - start,
        {},
    )


def simulated_annealing(
    graph: nx.DiGraph,
    model: QuboModel,
    block_count: int,
    iterations: int,
    seed: int,
) -> MethodResult:
    """실제 BFS 감염 수를 최소화하며 항상 K개를 유지하는 교환형 SA다."""
    start = time.perf_counter()
    rng = random.Random(seed)
    current = set(rng.sample(range(len(model.nodes)), block_count))
    current_remaining, _ = spread_counts_for_indices(graph, model, current)
    best, best_remaining = set(current), current_remaining

    for step in range(iterations):
        fraction = step / max(iterations - 1, 1)
        initial_temperature = max(1.0, len(model.baseline_infected) * 0.10)
        temperature = initial_temperature * (0.01 / initial_temperature) ** fraction
        removed = rng.choice(sorted(current))
        added = rng.choice(sorted(set(range(len(model.nodes))) - current))
        proposal = (current - {removed}) | {added}
        # proposal_remaining, _ = spread_counts_for_indices(graph, model, proposal)
        # delta = proposal_remaining - current_remaining
        # if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
        #     current, current_remaining = proposal, proposal_remaining
        #     if (
        #         current_remaining < best_remaining
        #         or (
        #             current_remaining == best_remaining
        #             and qubo_energy(model, current) < qubo_energy(model, best)
        #         )
        #     ):
        #         best, best_remaining = set(current), current_remaining
        proposal_energy = qubo_energy(model, proposal)
        current_energy = qubo_energy(model, current)
        delta = proposal_energy - current_energy
        
        if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
            current = proposal
            if (
                proposal_energy < qubo_energy(model, best)
                or (
                    math.isclose(proposal_energy, qubo_energy(model, best), abs_tol=1e-9)
                    and spread_counts_for_indices(graph, model, proposal)[0] < spread_counts_for_indices(graph, model, best)[0]
                )
            ):
                best = set(current)

    return MethodResult(
        "Simulated Annealing",
        tuple(sorted(best)),
        qubo_energy(model, best),
        time.perf_counter() - start,
        {"iterations": iterations, "seed": seed},
    )


def betweenness_selection(
    graph: nx.DiGraph,
    model: QuboModel,
    block_count: int,
) -> MethodResult:
    start = time.perf_counter()
    candidate_set = set(model.nodes)
    relevant_nodes = candidate_set | set(model.infection_sources)
    for node in model.nodes:
        for source in model.infection_sources:
            try:
                relevant_nodes.update(nx.shortest_path(graph, source, node))
            except nx.NetworkXNoPath:
                pass
    relevant_graph = graph.subgraph(relevant_nodes).copy()
    scores = nx.betweenness_centrality(relevant_graph, normalized=True)
    selected = sorted(
        range(len(model.nodes)),
        key=lambda i: (-scores.get(model.nodes[i], 0.0), model.nodes[i]),
    )[:block_count]
    return MethodResult(
        "Betweenness Centrality",
        tuple(sorted(selected)),
        qubo_energy(model, selected),
        time.perf_counter() - start,
        {"candidate_scores": {node: scores.get(node, 0.0) for node in model.nodes}},
    )


def qubo_to_ising(
    model: QuboModel,
) -> tuple[float, np.ndarray, dict[tuple[int, int], float]]:
    """x=(1-Z)/2를 사용해 QUBO를 Ising Z/ZZ 계수로 바꾼다."""
    ising_constant = (
        model.constant
        + 0.5 * float(np.sum(model.linear))
        + 0.25 * sum(model.quadratic.values())
    )
    z_fields = -0.5 * model.linear.copy()
    zz_couplings: dict[tuple[int, int], float] = {}
    for (i, j), coefficient in model.quadratic.items():
        z_fields[i] -= 0.25 * coefficient
        z_fields[j] -= 0.25 * coefficient
        zz_couplings[i, j] = 0.25 * coefficient
    return ising_constant, z_fields, zz_couplings


def qubo_hamiltonian(model: QuboModel) -> SparsePauliOp:
    """Aer Estimator가 계산할 QUBO 비용 해밀토니안을 만든다."""
    constant, z_fields, zz_couplings = qubo_to_ising(model)
    terms: list[tuple[str, list[int], complex]] = [("", [], constant)]
    terms.extend(
        ("Z", [qubit], coefficient)
        for qubit, coefficient in enumerate(z_fields)
        if abs(coefficient) > 1.0e-14
    )
    terms.extend(
        ("ZZ", [i, j], coefficient)
        for (i, j), coefficient in zz_couplings.items()
        if abs(coefficient) > 1.0e-14
    )
    return SparsePauliOp.from_sparse_list(
        terms,
        num_qubits=len(model.nodes),
    ).simplify()


def build_qaoa_circuit(
    model: QuboModel,
    reps: int,
    block_count: int,
) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """고정 Hamming weight 초기 상태와 ring XY mixer를 사용하는 QAOA 회로를 만든다."""
    _, z_fields, zz_couplings = qubo_to_ising(model)
    gamma = ParameterVector("gamma", reps)
    beta = ParameterVector("beta", reps)
    num_nodes = len(model.nodes)
    circuit = QuantumCircuit(num_nodes, name="blocking_qaoa")
    # circuit.h(range(num_nodes))

    # for layer in range(reps):
    #     for qubit, coefficient in enumerate(z_fields):
    #         if abs(coefficient) > 1.0e-14:
    #             circuit.rz(2.0 * gamma[layer] * coefficient, qubit)
    #     for (i, j), coefficient in zz_couplings.items():
    #         if abs(coefficient) > 1.0e-14:
    #             circuit.rzz(2.0 * gamma[layer] * coefficient, i, j)
    #     for qubit in range(len(model.nodes)):
    #         circuit.rx(2.0 * beta[layer], qubit)

    # return circuit, gamma, beta
    # Deterministic feasible initial state: the first K candidates are selected.
    # This is fixed-Hamming-weight initialization, not a Greedy warm start.
    for i in range(block_count):
        circuit.x(i)

    for layer in range(reps):
        # 2. 문제 해밀토니안 (비용 계산 - 기존과 동일)
        for qubit, coefficient in enumerate(z_fields):
            if abs(coefficient) > 1.0e-14:
                circuit.rz(2.0 * gamma[layer] * coefficient, qubit)
        for (i, j), coefficient in zz_couplings.items():
            if abs(coefficient) > 1.0e-14:
                    circuit.rzz(2.0 * gamma[layer] * coefficient, i, j)

        # 3. XY-믹서 적용 (Ring Topology)
        # 큐비트들을 원형(Ring)으로 이어주며 상태를 교환(Swap)시킵니다.
        # X_i X_j + Y_i Y_j 상호작용은 1의 총 개수(K)를 절대 깨뜨리지 않습니다.
        for i in range(num_nodes):
            j = (i + 1) % num_nodes # 원형으로 끝과 처음을 연결
            circuit.rxx(beta[layer], i, j)
            circuit.ryy(beta[layer], i, j)

    return circuit, gamma, beta


def aer_parameter_values(
    circuit: QuantumCircuit,
    gamma: ParameterVector,
    beta: ParameterVector,
    parameters: Sequence[float],
    reps: int,
) -> list[float]:
    """Aer Primitive이 요구하는 회로 파라미터 순서로 값을 정렬한다."""
    parameter_map = {
        **{gamma[i]: float(parameters[i]) for i in range(reps)},
        **{beta[i]: float(parameters[reps + i]) for i in range(reps)},
    }
    return [parameter_map[parameter] for parameter in circuit.parameters]


def index_to_selection(index: int, node_count: int) -> tuple[int, ...]:
    """Qiskit의 little-endian basis index를 선택된 x_i=1 인덱스로 바꾼다."""
    return tuple(i for i in range(node_count) if (index >> i) & 1)


def choose_feasible_sample(
    counts: dict[str, int],
    model: QuboModel,
    block_count: int,
) -> tuple[tuple[int, ...], int, int, int, bool]:
    """측정값 중 정확히 K개인 최소 비용 표본을 선택한다."""
    feasible_samples: list[tuple[float, int, int, tuple[int, ...]]] = []
    for bitstring, count in counts.items():
        basis_index = int(str(bitstring).replace(" ", ""), 2)
        selected = index_to_selection(basis_index, len(model.nodes))
        if len(selected) == block_count:
            feasible_samples.append(
                (qubo_energy(model, selected), -int(count), basis_index, selected)
            )

    fallback_used = False
    if feasible_samples:
        _, negative_count, chosen_index, selected = min(feasible_samples)
        chosen_count = -negative_count
    else:
        # 방어적으로 feasible sample이 하나도 없을 때만 가장 빈번한 표본을
        # 최소 에너지 방향으로 K개 선택 상태로 보정한다.
        fallback_used = True
        most_frequent = max(
            counts,
            key=lambda bitstring: (counts[bitstring], bitstring),
        )
        selected_set = set(
            index_to_selection(
                int(str(most_frequent).replace(" ", ""), 2),
                len(model.nodes),
            )
        )
        while len(selected_set) > block_count:
            removed = min(
                selected_set,
                key=lambda node: qubo_energy(model, selected_set - {node}),
            )
            selected_set.remove(removed)
        while len(selected_set) < block_count:
            remaining = set(range(len(model.nodes))) - selected_set
            added = min(
                remaining,
                key=lambda node: qubo_energy(model, selected_set | {node}),
            )
            selected_set.add(added)
        selected = tuple(sorted(selected_set))
        chosen_index = sum(1 << index for index in selected)
        chosen_count = int(counts[most_frequent])

    return (
        tuple(selected),
        chosen_index,
        chosen_count,
        len(feasible_samples),
        fallback_used,
    )


def _optional_float(value: object) -> float | None:
    """IBM Runtime 메트릭의 숫자/문자열 값을 안전하게 초 단위 실수로 바꾼다."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_ibm_job_times(job: object, metrics: object) -> dict:
    """
    IBM Runtime의 버전별 반환 형식을 모두 고려해 QPU 시간을 분리한다.

    - qaoa_qpu_circuit_seconds: QAOA 회로가 shots만큼 실제 실행된 누적 시간
    - qpu_usage_seconds: 준비 오버헤드까지 포함해 QPU가 이 job에 잠긴 할당량 시간

    2026-04 REST 형식은 circuits_execution_time_ns를 metrics 최상위에 두고,
    이전 형식은 usage 내부에 둔다. job.usage()는 최신 Runtime의 할당량 시간이다.
    """
    metric_dict = metrics if isinstance(metrics, dict) else {}
    usage_metrics = metric_dict.get("usage", {})
    if not isinstance(usage_metrics, dict):
        usage_metrics = {}

    circuit_ns = _optional_float(metric_dict.get("circuits_execution_time_ns"))
    circuit_time_source = "metrics.circuits_execution_time_ns"
    if circuit_ns is None:
        circuit_ns = _optional_float(
            usage_metrics.get("circuits_execution_time_ns")
        )
        circuit_time_source = "metrics.usage.circuits_execution_time_ns"

    circuit_seconds = circuit_ns / 1_000_000_000.0 if circuit_ns is not None else None

    qpu_usage_seconds = None
    usage_method = getattr(job, "usage", None)
    if callable(usage_method):
        try:
            qpu_usage_seconds = _optional_float(usage_method(partial=False))
        except TypeError:
            # qiskit-ibm-runtime 0.46 등 partial 인자가 없는 버전도 지원한다.
            qpu_usage_seconds = _optional_float(usage_method())
        except Exception:
            # 아래의 이전 metrics 형식으로 다시 시도한다.
            qpu_usage_seconds = None

    if qpu_usage_seconds is None:
        qpu_usage_seconds = _optional_float(usage_metrics.get("quantum_seconds"))

    if qpu_usage_seconds is None:
        usage_estimation = getattr(job, "usage_estimation", None)
        if callable(usage_estimation):
            usage_estimation = usage_estimation()
        if isinstance(usage_estimation, dict):
            qpu_usage_seconds = _optional_float(
                usage_estimation.get("quantum_seconds")
            )

    # 사용자가 요청한 표의 QAOA 시간은 순수 회로 실행시간을 최우선으로 한다.
    # 구형 Runtime이 회로 시간을 주지 않으면 QPU 사용시간을 대체값으로 쓴다.
    if circuit_seconds is not None:
        reported_seconds = circuit_seconds
        reported_source = circuit_time_source
    elif qpu_usage_seconds is not None:
        reported_seconds = qpu_usage_seconds
        reported_source = "job.usage() fallback (includes QPU preparation overhead)"
    else:
        reported_seconds = None
        reported_source = "unavailable"

    return {
        "qaoa_qpu_circuit_seconds": circuit_seconds,
        "qaoa_qpu_circuit_time_source": (
            circuit_time_source if circuit_seconds is not None else None
        ),
        "qpu_usage_seconds": qpu_usage_seconds,
        "reported_qaoa_runtime_seconds": reported_seconds,
        "reported_qaoa_runtime_source": reported_source,
    }


def run_ibm_sampler(
    circuit: QuantumCircuit,
    gamma: ParameterVector,
    beta: ParameterVector,
    optimized_parameters: Sequence[float],
    reps: int,
    shots: int,
    node_count: int,
    backend_name: str | None,
    dd_sequence: str,
    max_execution_time: int,
) -> tuple[dict[str, int], dict]:
    """Aer에서 최적화한 최종 회로 한 건을 IBM QPU의 job mode로 실행한다."""
    try:
        import qiskit_ibm_runtime
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )
        from qiskit_ibm_runtime import (
            QiskitRuntimeService,
            SamplerV2 as RuntimeSamplerV2,
        )
    except ImportError as error:
        raise RuntimeError(
            "IBM QPU 실행에는 qiskit-ibm-runtime이 필요합니다. "
            "'python -m pip install -U qiskit-ibm-runtime'으로 설치하세요."
        ) from error

    service = QiskitRuntimeService()
    if backend_name:
        backend = service.backend(name=backend_name)
        if not backend.status().operational:
            raise RuntimeError(f"IBM QPU가 현재 operational 상태가 아닙니다: {backend_name}")
        if backend.num_qubits < node_count:
            raise ValueError(
                f"{backend_name}은 {backend.num_qubits}큐빗이므로 "
                f"{node_count}큐빗 회로를 실행할 수 없습니다."
            )
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=node_count,
        )

    parameter_map = {
        **{gamma[i]: float(optimized_parameters[i]) for i in range(reps)},
        **{
            beta[i]: float(optimized_parameters[reps + i])
            for i in range(reps)
        },
    }
    hardware_circuit = circuit.assign_parameters(parameter_map)
    hardware_circuit.measure_all()

    transpile_start = time.perf_counter()
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
    )
    isa_circuit = pass_manager.run(hardware_circuit)
    transpile_seconds = time.perf_counter() - transpile_start
    operation_counts = {
        str(name): int(count) for name, count in isa_circuit.count_ops().items()
    }

    print(f"Selected IBM QPU: {backend.name} ({backend.num_qubits} qubits)")
    print(
        "Transpiled circuit: "
        f"depth={isa_circuit.depth():,}, operations={operation_counts}"
    )

    sampler = RuntimeSamplerV2(mode=backend)
    sampler.options.default_shots = shots
    sampler.options.max_execution_time = max_execution_time
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = dd_sequence
    sampler.options.environment.job_tags = ["quantum-reframing-qaoa"]

    submitted_at = time.perf_counter()
    job = sampler.run([isa_circuit], shots=shots)
    print(f"IBM Job ID: {job.job_id()}")
    print("QPU 대기열에 제출했습니다. 완료될 때까지 이 창을 닫지 마세요.")
    hardware_result = job.result()
    job_wall_seconds = time.perf_counter() - submitted_at
    counts = hardware_result[0].data.meas.get_counts()
    metrics = job.metrics()
    json_metrics = json.loads(json.dumps(metrics, default=str))
    ibm_times = extract_ibm_job_times(job, metrics)

    return counts, {
        "qiskit_ibm_runtime_version": qiskit_ibm_runtime.__version__,
        "backend": backend.name,
        "backend_qubits": int(backend.num_qubits),
        "execution_mode": "job",
        "job_id": job.job_id(),
        "job_wall_seconds_including_queue": job_wall_seconds,
        **ibm_times,
        "job_metrics": json_metrics,
        "transpile_seconds": transpile_seconds,
        "transpiled_depth": int(isa_circuit.depth()),
        "transpiled_operations": operation_counts,
        "transpile_optimization_level": 3,
        "dynamical_decoupling": dd_sequence,
        "max_execution_time": max_execution_time,
    }


def qaoa(
    model: QuboModel,
    block_count: int,
    reps: int,
    maxiter: int,
    restarts: int,
    shots: int,
    seed: int,
    execution_backend: str,
    ibm_backend_name: str | None,
    ibm_dd_sequence: str,
    ibm_max_execution_time: int,
) -> MethodResult:
    start = time.perf_counter()
    optimization_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    best_optimization = None
    hamiltonian = qubo_hamiltonian(model)
    circuit, gamma, beta = build_qaoa_circuit(model, reps, block_count)
    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "method": "matrix_product_state",
                "device": "CPU",
                "matrix_product_state_max_bond_dimension": 64,
                "matrix_product_state_truncation_threshold": 1.0e-8,
            },
            "run_options": {"seed_simulator": seed},
        }
    )

    def expectation(parameters: np.ndarray) -> float:
        values = aer_parameter_values(circuit, gamma, beta, parameters, reps)
        result = estimator.run([(circuit, hamiltonian, values)]).result()
        return float(np.asarray(result[0].data.evs).item())

    for _ in range(restarts):
        initial = np.concatenate(
            [rng.uniform(0.0, math.pi, reps), rng.uniform(0.0, math.pi / 2.0, reps)]
        )
        result = minimize(
            expectation,
            initial,
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1.0e-3},
        )
        if best_optimization is None or result.fun < best_optimization.fun:
            best_optimization = result

    if best_optimization is None:
        raise RuntimeError("QAOA 파라미터 최적화에 실패했습니다.")

    optimization_seconds = time.perf_counter() - optimization_start
    execution_details: dict
    if execution_backend == "aer":
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()
        final_values = aer_parameter_values(
            measured_circuit,
            gamma,
            beta,
            best_optimization.x,
            reps,
        )
        sampler = AerSamplerV2(
            default_shots=shots,
            seed=seed + 1,
            options={
                "backend_options": {
                    "method": "matrix_product_state",
                    "device": "CPU",
                    "matrix_product_state_max_bond_dimension": 64,
                    "matrix_product_state_truncation_threshold": 1.0e-8,
                }
            },
        )
        sample_result = sampler.run(
            [(measured_circuit, final_values)],
            shots=shots,
        ).result()
        counts = sample_result[0].data.meas.get_counts()
        method_name = "QAOA (Qiskit Aer)"
        execution_details = {
            "qiskit_aer_version": qiskit_aer.__version__,
            "backend": "Aer EstimatorV2 + SamplerV2",
            "simulation_method": "matrix_product_state",
            "device": "CPU",
            "expectation_mode": "MPS approximation with bond dimension capped at 64",
        }
    else:
        counts, execution_details = run_ibm_sampler(
            circuit,
            gamma,
            beta,
            best_optimization.x,
            reps,
            shots,
            len(model.nodes),
            ibm_backend_name,
            ibm_dd_sequence,
            ibm_max_execution_time,
        )
        method_name = "QAOA (IBM QPU, Aer-optimized)"

    (
        selected,
        chosen_index,
        chosen_count,
        feasible_sample_count,
        fallback_used,
    ) = choose_feasible_sample(counts, model, block_count)

    hybrid_end_to_end_seconds = time.perf_counter() - start
    if execution_backend == "ibm":
        reported_qaoa_seconds = execution_details[
            "reported_qaoa_runtime_seconds"
        ]
        if reported_qaoa_seconds is None:
            # 결과 파일을 버리지 않되, wall time을 QPU 시간인 것처럼 표시하지 않는다.
            reported_qaoa_seconds = math.nan
        runtime = float(reported_qaoa_seconds)
    else:
        runtime = hybrid_end_to_end_seconds
    return MethodResult(
        method_name,
        tuple(selected),
        qubo_energy(model, selected),
        runtime,
        {
            "qiskit_version": qiskit.__version__,
            **execution_details,
            "parameter_optimization_backend": "Qiskit Aer MPS/CPU",
            "parameter_optimization_seconds": optimization_seconds,
            "hybrid_end_to_end_seconds": hybrid_end_to_end_seconds,
            "reps": reps,
            "optimizer": "COBYLA",
            "optimizer_evaluations": int(best_optimization.nfev),
            "optimized_expectation": float(best_optimization.fun),
            "optimized_parameters": [float(value) for value in best_optimization.x],
            "shots": shots,
            "chosen_sample_count": chosen_count,
            "feasible_samples_observed": feasible_sample_count,
            "fallback_cardinality_repair_used": fallback_used,
            "basis_index": chosen_index,
        },
    )


def evaluate_results(
    graph: nx.DiGraph,
    model: QuboModel,
    results: Sequence[MethodResult],
    oracle: SpreadOracle | None,
) -> list[dict]:
    """모든 방법을 실제 재확산 후 감염 감소 패키지 수로 평가한다."""
    baseline_count = len(model.baseline_infected)
    rows: list[dict] = []
    for result in results:
        remaining_count, reduction_count = spread_counts_for_indices(
            graph,
            model,
            result.selected,
        )
        block_count = len(result.selected)
        if oracle is None:
            gap_to_oracle: int | None = None
            oracle_efficiency: float | None = None
        else:
            gap_to_oracle = oracle.infection_reduction_count - reduction_count
            oracle_efficiency = (
                reduction_count / oracle.infection_reduction_count
                if oracle.infection_reduction_count > 0
                else 1.0
            )
        if oracle is None:
            qubo_efficiency = None
            qubo_gap = None
        else:
            # 이 방법의 QUBO 에너지가 QUBO 최적 대비 얼마나 가까운가
            # (에너지는 낮을수록 좋으므로, 최적/현재 비율 대신 차이로)
            qubo_gap = result.objective - oracle.best_qubo_energy
            # 정규화: QUBO 최적을 얼마나 달성했나 (1.0이면 QUBO 최적 도달)
            qubo_efficiency = (
                oracle.best_qubo_energy / result.objective
                if result.objective != 0 else None
            )
        rows.append(
            {
                "method": result.method,
                "baseline_infected_count": baseline_count,
                "remaining_infected_count": remaining_count,
                "infection_reduction_count": reduction_count,
                "infection_reduction_ratio": (
                    reduction_count / baseline_count if baseline_count else 0.0
                ),
                "reduction_per_block": (
                    reduction_count / block_count if block_count else 0.0
                ),
                "gap_to_exact_reduction": gap_to_oracle,
                "efficiency_vs_exact_reduction": oracle_efficiency,
                "qubo_efficiency": qubo_efficiency,      
                "qubo_gap_to_best": qubo_gap, 
                "qubo_objective": result.objective,
                "runtime_seconds": result.runtime_seconds,
                "selected_nodes": json.dumps(
                    [model.nodes[index] for index in result.selected],
                    ensure_ascii=False,
                ),
                "details": result.details,
            }
        )

    # 2단계의 주평가지표는 감염 감소량이다. QUBO 값은 동률 해소/감사용이다.
    rows.sort(
        key=lambda row: (
            row["qubo_objective"],
            -row["infection_reduction_count"],
            row["remaining_infected_count"],
            (
                row["runtime_seconds"]
                if math.isfinite(row["runtime_seconds"])
                else math.inf
            ),
            row["method"],
        )
    )
    return [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]


def write_outputs(
    output_dir: Path,
    model: QuboModel,
    oracle: SpreadOracle | None,
    ranked_results: Sequence[dict],
    stage1_pool: Sequence[str],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "method_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "rank",
            "method",
            "baseline_infected_count",
            "remaining_infected_count",
            "infection_reduction_count",
            "infection_reduction_ratio",
            "reduction_per_block",
            "gap_to_exact_reduction",
            "efficiency_vs_exact_reduction",
            "qubo_objective",
            "runtime_seconds",
            "selected_nodes",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                key: row[key]
                for key in fieldnames
            }
            for row in ranked_results
        )

    selected_path = output_dir / "selected_nodes.csv"
    with selected_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "method",
                "node",
                "selected_for_blocking",
                "single_block_infection_reduction_count",
            ],
        )
        writer.writeheader()
        for row in ranked_results:
            for node in json.loads(row["selected_nodes"]):
                writer.writerow(
                    {
                        "rank": row["rank"],
                        "method": row["method"],
                        "node": node,
                        "selected_for_blocking": 1,
                        "single_block_infection_reduction_count": int(
                            model.single_block_reductions[model.nodes.index(node)]
                        ),
                    }
                )

    with (output_dir / "qubo_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fieldnames = [
            "stage1_rank",
            "node",
            "used_as_qubo_candidate",
            "single_block_infection_reduction_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        qubo_set = set(model.nodes)
        qubo_index = {node: i for i, node in enumerate(model.nodes)}
        for rank, node in enumerate(stage1_pool, start=1):
            index = qubo_index.get(node)
            writer.writerow(
                {
                    "stage1_rank": rank,
                    "node": node,
                    "used_as_qubo_candidate": int(node in qubo_set),
                    "single_block_infection_reduction_count": (
                        int(model.single_block_reductions[index])
                        if index is not None
                        else ""
                    ),
                }
            )

    report = {
        "problem": {
            "stage1_candidate_count": len(stage1_pool),
            "qubo_candidate_count": len(model.nodes),
            "block_count": args.block_count,
            "candidate_source": str(args.rankings.resolve()),
            "candidate_method": args.candidate_method,
            "infection_sources": list(model.infection_sources),
            "baseline_infected_count": len(model.baseline_infected),
            "qaoa_execution_backend": args.qaoa_backend,
        },
        "data_leakage_guard": {
            "infection_label_read_or_used": False,
            "stage1_risk_score_used_by_stage2": False,
            "known_infection_source_is_input": True,
        },
        "qubo": {
            "objective": (
                "-single_block_reduction - pair_interaction + "
                "availability_cost_weight * normalized_in_degree"
            ),
            "definition": (
                "Pairwise approximation of actual BFS infection-count reduction; "
                "final ranking always uses a full BFS re-simulation."
            ),
            "availability_cost_weight": args.availability_cost_weight,
            "cardinality_constraint": (
                "exactly K selected by fixed-Hamming-weight initialization "
                "and XY mixer; no cardinality penalty term"
            ),
            "linear": model.linear.tolist(),
            "quadratic": {
                f"{i},{j}": value for (i, j), value in model.quadratic.items()
            },
            "constant": model.constant,
        },
        "candidates": [
            {
                "index": i,
                "node": node,
                "single_block_infection_reduction_count": int(
                    model.single_block_reductions[i]
                ),
                "prevented_nodes_if_blocked_alone": len(
                    model.prevented_by_single_block[node]
                ),
            }
            for i, node in enumerate(model.nodes)
        ],
        "exact_spread_oracle": (
            {
                "available": True,
                "selected_nodes": [model.nodes[i] for i in oracle.selected],
                "remaining_infected_count": oracle.remaining_infected_count,
                "infection_reduction_count": oracle.infection_reduction_count,
                "runtime_seconds": oracle.runtime_seconds,
                "combinations_checked": oracle.combinations_checked,
            }
            if oracle is not None
            else {
                "available": False,
                "reason": "combination count exceeded --exact-max-combinations",
            }
        ),
        "ranking_rule": (
            "QUBO objective ascending, infection_reduction_count descending, "
            "remaining_infected_count ascending, runtime ascending"
        ),
        "results": [
            {**row, "selected_nodes": json.loads(row["selected_nodes"])}
            for row in ranked_results
        ],
        "notes": [
            (
                f"Stage 1 provides {len(stage1_pool)} candidates; only its top "
                f"{len(model.nodes)} node names enter Stage 2, and exactly "
                f"{args.block_count} are blocked."
            ),
            "No classification metric is used anywhere in Stage 2.",
            "JSON infection_label and ranking risk_score are not read by Stage 2.",
            "The known infection source is scenario input, not a hidden target label.",
            "All method rankings are based on the exact post-block BFS infection count.",
            (
                "QAOA parameters are optimized with Qiskit Aer; the final bound circuit is transpiled and sampled on an IBM QPU."
                if args.qaoa_backend == "ibm"
                else "QAOA uses Qiskit Aer MPS simulation with bond dimension capped at 64. This is an approximation."
            ),
            (
                "For the IBM row, runtime_seconds reports only the final QAOA circuit execution time on the QPU when IBM provides circuits_execution_time_ns. It excludes Aer parameter optimization, transpilation, queueing, and classical post-processing."
                if args.qaoa_backend == "ibm"
                else "Aer runs an approximate MPS simulation on the local CPU, so runtime does not demonstrate quantum speedup."
            ),
            (
                "This IBM mode does not run the full hybrid QAOA optimization loop on the QPU: parameters are optimized on Aer and only the final bound QAOA circuit is sampled on hardware."
                if args.qaoa_backend == "ibm"
                else ""
            ),
            "The QUBO uses single-node and pairwise BFS effects; higher-order effects are captured only by final full-BFS evaluation.",
        ],
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def print_results(
    model: QuboModel,
    oracle: SpreadOracle | None,
    ranked_results: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    print(f"Stage 1 candidates loaded: {args.stage1_candidate_count}")
    print(f"QUBO/QAOA candidates used: {len(model.nodes)}")
    print(f"Nodes to block: {args.block_count}")
    print(f"Known infection sources: {', '.join(model.infection_sources)}")
    print(f"Baseline infected packages: {len(model.baseline_infected)}")
    print("infection_label/risk_score input leakage: blocked (not read)")
    if args.qaoa_backend == "ibm":
        print(
            "QAOA execution: Aer parameter optimization -> IBM QPU final sampling, "
            f"{len(model.nodes)} qubits"
        )
    else:
        print(
            "QAOA simulation: "
            f"Qiskit {qiskit.__version__}, Aer {qiskit_aer.__version__}, "
            f"EstimatorV2 + SamplerV2, MPS/CPU, {len(model.nodes)} qubits"
        )
    combination_count = math.comb(len(model.nodes), args.block_count)
    if oracle is None:
        print(
            f"Exact spread oracle: skipped; C({len(model.nodes)}, "
            f"{args.block_count})={combination_count:,} exceeds limit"
        )
    else:
        print(
            f"Exact spread oracle: {oracle.infection_reduction_count} packages "
            f"reduced after checking {oracle.combinations_checked:,} combinations"
        )
    print()
    print(
        f"{'rank':>4}  {'method':28} "
        f"{'reduced':>8} {'BFS_최적%':>9} "    # 실제 성능
        f"{'QUBOE':>10} {'QUBO최적%':>10} "    # 근사 성능
        f"{'time(s)':>10}"
    )
    for row in ranked_results:
        bfs_eff = row["efficiency_vs_exact_reduction"]
        qubo_eff = row["qubo_efficiency"]
        bfs_text = f"{bfs_eff*100:.0f}%" if bfs_eff is not None else "-"
        qubo_text = f"{qubo_eff*100:.0f}%" if qubo_eff is not None else "-"
        print(
            f"{row['rank']:>4}  {row['method'][:28]:28} "
            f"{row['infection_reduction_count']:8d} {bfs_text:>9} "
            f"{row['qubo_objective']:10.3f} {qubo_text:>10} "
            f"{row['runtime_seconds']:10.4f}"
        )

    if args.qaoa_backend == "ibm":
        qaoa_row = next(
            row for row in ranked_results if row["method"].startswith("QAOA")
        )
        details = qaoa_row["details"]
        print("\nIBM QPU execution details:")
        print(f"  backend: {details['backend']}")
        print(f"  job id: {details['job_id']}")
        print(f"  transpiled depth: {details['transpiled_depth']:,}")
        qaoa_circuit_seconds = details["qaoa_qpu_circuit_seconds"]
        if qaoa_circuit_seconds is None:
            print("  QAOA QPU circuit time: unavailable from IBM metrics")
        else:
            print(
                "  QAOA QPU circuit time (requested): "
                f"{qaoa_circuit_seconds:.9f}s"
            )
        print(
            "  table time source: "
            f"{details['reported_qaoa_runtime_source']}"
        )
        qpu_usage_seconds = details["qpu_usage_seconds"]
        if qpu_usage_seconds is None:
            print("  QPU locked usage (quota): unavailable")
        else:
            print(f"  QPU locked usage (quota): {qpu_usage_seconds:.6f}s")
        print(
            "  Aer parameter optimization: "
            f"{details['parameter_optimization_seconds']:.3f}s"
        )
        print(f"  transpilation: {details['transpile_seconds']:.3f}s")
        print(
            "  wall time including queue: "
            f"{details['job_wall_seconds_including_queue']:.3f}s"
        )
        print(
            "  hybrid end-to-end: "
            f"{details['hybrid_end_to_end_seconds']:.3f}s"
        )
    print("\nSelected blocking nodes:")
    for row in ranked_results:
        print(f"\n{row['rank']}. {row['method']}")
        for node in json.loads(row["selected_nodes"]):
            print(f"  - {node}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1 랭킹에서 상위 후보를 가져와 정확히 K개를 차단하고, "
            "차단 후 감염 패키지 수 감소량을 최적화합니다. infection_label은 "
            "Stage 2의 입력으로 사용하지 않습니다."
        )
    )
    parser.add_argument("--data", type=Path, default=default_data_path())
    parser.add_argument("--rankings", type=Path, default=default_rankings_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--infected",
        action="append",
        default=None,
        help=(
            "알려진 최초 감염원입니다. 여러 개면 옵션을 반복하세요. "
            "기본값: es-errors@1.3.0"
        ),
    )
    parser.add_argument("--candidate-method", default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument(
        "--stage1-candidate-count", type=int, default=DEFAULT_STAGE1_CANDIDATE_COUNT
    )
    parser.add_argument(
        "--candidate-count", type=int, default=DEFAULT_QUBO_CANDIDATE_COUNT
    )
    parser.add_argument("--block-count", type=int, default=DEFAULT_BLOCK_COUNT)
    parser.add_argument(
        "--availability-cost-weight",
        type=float,
        default=DEFAULT_AVAILABILITY_COST_WEIGHT,
        help=(
            "방어 효과 대비 정상 서비스 영향 비용의 가중치 "
            f"(default: {DEFAULT_AVAILABILITY_COST_WEIGHT})."
        ),
    )
    parser.add_argument(
        "--exact-max-combinations",
        type=int,
        default=200_000_000,
        help="실제 BFS 정확해를 계산할 최대 조합 수입니다.",
    )
    parser.add_argument("--sa-iterations", type=int, default=20_000)
    parser.add_argument("--qaoa-reps", type=int, default=1)
    parser.add_argument("--qaoa-maxiter", type=int, default=100)
    parser.add_argument("--qaoa-restarts", type=int, default=5)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--qaoa-backend",
        choices=["aer", "ibm"],
        default="aer",
        help="QAOA 최종 회로 실행 위치입니다. 기본값은 aer입니다.",
    )
    parser.add_argument(
        "--ibm-backend",
        default=None,
        help="실행할 IBM QPU 이름입니다. 생략하면 사용 가능한 장비 중 least busy를 고릅니다.",
    )
    parser.add_argument(
        "--ibm-dd-sequence",
        choices=["XX", "XpXm", "XY4"],
        default="XY4",
        help="IBM QPU dynamical decoupling 시퀀스입니다.",
    )
    parser.add_argument(
        "--ibm-max-execution-time",
        type=int,
        default=600,
        help="IBM QPU 작업의 최대 실행시간(초)입니다.",
    )
    parser.add_argument(
        "--confirm-qpu",
        action="store_true",
        help="실제 IBM QPU 할당량 사용에 동의하고 작업 제출을 허용합니다.",
    )
    parser.add_argument("--seed", type=int, default=2_026)
    args = parser.parse_args()
    if args.infected is None:
        args.infected = ["es-errors@1.3.0"]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.data.is_file():
        raise ValueError(f"데이터 파일이 없습니다: {args.data}")
    if not args.rankings.is_file():
        raise ValueError(f"1단계 rankings.csv가 없습니다: {args.rankings}")
    if args.stage1_candidate_count <= 0:
        raise ValueError("1단계 후보 수는 양수여야 합니다.")
    if not 0 < args.candidate_count <= args.stage1_candidate_count:
        raise ValueError("2단계 후보 수는 1 이상이며 1단계 후보 수 이하여야 합니다.")
    if not 0 < args.block_count < args.candidate_count:
        raise ValueError("차단 수는 0보다 크고 후보 수보다 작아야 합니다.")
    if args.availability_cost_weight < 0.0:
        raise ValueError("availability cost weight는 0 이상이어야 합니다.")
    if args.exact_max_combinations <= 0:
        raise ValueError("exact max combinations는 양수여야 합니다.")
    if not args.infected or any(not item.strip() for item in args.infected):
        raise ValueError("최초 감염원은 하나 이상 필요합니다.")
    if min(
        args.sa_iterations,
        args.qaoa_reps,
        args.qaoa_maxiter,
        args.qaoa_restarts,
        args.shots,
    ) <= 0:
        raise ValueError("반복 횟수와 shots는 양수여야 합니다.")
    if args.ibm_max_execution_time <= 0:
        raise ValueError("IBM max execution time은 양수여야 합니다.")
    if args.qaoa_backend == "ibm" and not args.confirm_qpu:
        raise ValueError(
            "실제 IBM QPU 할당량을 사용합니다. 실행하려면 "
            "'--qaoa-backend ibm --confirm-qpu'를 함께 지정하세요."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    graph = load_propagation_graph(args.data)
    stage1_pool, candidates = load_stage1_candidates(
        args.rankings,
        args.candidate_method,
        args.stage1_candidate_count,
        args.candidate_count,
    )
    model = build_qubo(
        graph,
        candidates,
        args.infected,
        args.block_count,
        args.availability_cost_weight,
    )

    oracle = exact_spread_oracle(
        graph,
        model,
        args.block_count,
        args.exact_max_combinations,
    )
    results = [
        qaoa(
            model,
            args.block_count,
            args.qaoa_reps,
            args.qaoa_maxiter,
            args.qaoa_restarts,
            args.shots,
            args.seed,
            args.qaoa_backend,
            args.ibm_backend,
            args.ibm_dd_sequence,
            args.ibm_max_execution_time,
        ),
        greedy(graph, model, args.block_count),
        simulated_annealing(
            graph,
            model,
            args.block_count,
            args.sa_iterations,
            args.seed,
        ),
        betweenness_selection(graph, model, args.block_count),
    ]

    ranked_results = evaluate_results(graph, model, results, oracle)
    write_outputs(
        args.output_dir,
        model,
        oracle,
        ranked_results,
        stage1_pool,
        args,
    )
    print_results(model, oracle, ranked_results, args)
    print(f"\nSaved outputs to: {args.output_dir.resolve()}")
    


if __name__ == "__main__":
    main()
