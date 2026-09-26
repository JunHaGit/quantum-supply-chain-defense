from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.linalg import expm

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler


# ============================================================
# Default experiment configuration
# ============================================================

DEFAULT_DATA_PATH = Path("data/gatsby_5.16.1_dependencies_labeled.json")
DEFAULT_INFECTED = "es-errors@1.3.0"

# The known infected source is included in the reduced CTQW subgraph.
# Therefore, candidate_limit=16 produces at most 17 modeled nodes.
DEFAULT_CANDIDATE_LIMIT = 16

DEFAULT_T_OBS = 300.0
DEFAULT_TROTTER_REPS = 10
DEFAULT_SHOTS = 10_000

DEFAULT_CTQW_TIME_START = 0.0
DEFAULT_CTQW_TIME_END = 5.0
DEFAULT_CTQW_TIME_STEPS = 200
DEFAULT_TOP_K = 15

DEFAULT_OUTPUT_CSV = Path("phase1_ctqw_candidates.csv")


def load_dependency_subgraph(
    data_path: Path,
    infected_package: str,
    candidate_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """
    Build a reduced infection-propagation graph from the deps.dev dependency data.

    deps.dev dependency edge:
        A -> B  : package A depends on package B

    Assumed infection-propagation direction:
        B -> A  : if B is compromised, A may be exposed

    The known infected source and up to `candidate_limit` reachable packages are
    retained for the reduced CTQW experiment.
    """
    with data_path.open("r", encoding="utf-8") as file:
        json_data = json.load(file)

    nodes = json_data["nodes"]
    edges = json_data["edges"]
    package_names = [
        f"{node['versionKey']['name']}@{node['versionKey']['version']}"
        for node in nodes
    ]

    if infected_package not in package_names:
        raise ValueError(
            f"감염원 패키지가 데이터에 없습니다: {infected_package}"
        )

    infected_source = package_names.index(infected_package)

    propagation_graph = nx.DiGraph()
    propagation_graph.add_nodes_from(range(len(nodes)))
    propagation_graph.add_edges_from(
        (edge["toNode"], edge["fromNode"])
        for edge in edges
    )

    reachable_indices = list(
        nx.bfs_tree(propagation_graph, infected_source).nodes()
    )
    if not reachable_indices:
        raise ValueError("감염원에서 도달 가능한 노드를 찾지 못했습니다.")

    # bfs_tree returns the source first, followed by reachable packages.
    selected_indices = reachable_indices[: candidate_limit + 1]
    selected_position = {
        original_index: local_index
        for local_index, original_index in enumerate(selected_indices)
    }

    adjacency = np.zeros(
        (len(selected_indices), len(selected_indices)),
        dtype=float,
    )

    for edge in edges:
        source = edge["toNode"]
        target = edge["fromNode"]
        if source in selected_position and target in selected_position:
            adjacency[
                selected_position[source],
                selected_position[target],
            ] = 1.0

    num_original_nodes = len(adjacency)
    num_qubits = max(1, int(np.ceil(np.log2(num_original_nodes))))
    total_states = 2**num_qubits

    padded_adjacency = np.zeros((total_states, total_states), dtype=float)
    padded_adjacency[:num_original_nodes, :num_original_nodes] = adjacency

    # Symmetric Hamiltonian used by this reduced CTQW hardware demo.
    hamiltonian = padded_adjacency + padded_adjacency.T

    initial_state = np.zeros(total_states, dtype=np.complex128)
    infected_local_index = selected_position[infected_source]
    initial_state[infected_local_index] = 1.0

    node_labels = [package_names[index] for index in selected_indices]
    node_labels.extend(
        f"Dummy {index}"
        for index in range(num_original_nodes, total_states)
    )

    print(
        f"데이터 로드 완료: 총 노드 {len(nodes)}개, "
        f"간선 {len(edges)}개"
    )
    print(f"감염원: {infected_package}")
    print(
        f"도달 가능 노드: {max(0, len(reachable_indices) - 1)}개, "
        f"CTQW 실험 노드: {num_original_nodes}개"
    )
    print(f"큐비트 수: {num_qubits}개")

    return (
        padded_adjacency,
        hamiltonian,
        initial_state,
        node_labels,
        num_original_nodes,
    )


def build_ctqw_circuit(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    num_qubits: int,
    t_obs: float,
    trotter_reps: int,
) -> QuantumCircuit:
    """Build the Trotterized CTQW circuit without submitting it."""
    circuit = QuantumCircuit(num_qubits)
    circuit.initialize(initial_state, range(num_qubits))

    pauli_op = SparsePauliOp.from_operator(hamiltonian)
    evolution_gate = PauliEvolutionGate(pauli_op, time=t_obs)

    synthesis = LieTrotter(reps=trotter_reps)
    trotterized_circuit = synthesis.synthesize(evolution_gate)
    circuit.compose(trotterized_circuit, inplace=True)
    circuit.measure_all()

    return circuit


def run_ctqw_circuit(
    circuit: QuantumCircuit,
    num_qubits: int,
    shots: int,
    use_hardware: bool,
) -> tuple[dict[str, int], str]:
    """
    Execute the circuit.

    Simulator mode is the default and requires no IBM Quantum credentials.
    Hardware mode uses credentials previously configured on the local machine;
    no API key or instance identifier is stored in this source file.
    """
    if not use_hardware:
        simulator = AerSimulator()
        compiled_circuit = transpile(circuit, simulator)
        result = simulator.run(
            compiled_circuit,
            shots=shots,
        ).result()
        counts = result.get_counts(compiled_circuit)
        return counts, "AerSimulator"

    # Credentials must be configured separately on the user's machine.
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=num_qubits,
    )
    print(f"\n[{backend.name}] IBM Quantum hardware에 작업을 전송합니다...")

    compiled_circuit = transpile(circuit, backend)
    sampler = Sampler(mode=backend)
    job = sampler.run([compiled_circuit], shots=shots)
    print(f"Job ID: {job.job_id()}")

    result = job.result()
    counts = result[0].data.meas.get_counts()
    return counts, backend.name


def print_measurement_probabilities(
    counts: dict[str, int],
    num_qubits: int,
    shots: int,
    backend_name: str,
) -> None:
    """Print measured probabilities for every computational basis state."""
    print(f"\n[{backend_name}] 측정 확률")

    for index in range(2**num_qubits):
        state = format(index, f"0{num_qubits}b")
        probability = counts.get(state, 0) / shots
        print(f"Node {index} (|{state}>) : {probability * 100:.2f}%")


def calculate_pagerank(
    adjacency: np.ndarray,
    alpha: float = 0.85,
) -> np.ndarray:
    """Calculate classical PageRank scores for comparison."""
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    scores = nx.pagerank(graph, alpha=alpha)
    return np.array(
        [scores[index] for index in range(len(adjacency))],
        dtype=float,
    )


def calculate_ctqw_max_peaks(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    t_start: float,
    t_end: float,
    num_steps: int,
) -> np.ndarray:
    """Calculate each state's maximum CTQW probability over a time interval."""
    time_steps = np.linspace(t_start, t_end, num_steps)
    probabilities = []

    for current_time in time_steps:
        evolution = expm(-1j * hamiltonian * current_time)
        state = evolution @ initial_state
        probabilities.append(np.abs(state) ** 2)

    return np.max(np.asarray(probabilities), axis=0)


def plot_comparison(
    labels: list[str],
    classical_scores: np.ndarray,
    quantum_scores: np.ndarray,
    top_k: int,
) -> None:
    """Compare PageRank and CTQW peak scores for the CTQW-ranked top-K nodes."""
    labels_array = np.asarray(labels)
    classical_scores = np.asarray(classical_scores)
    quantum_scores = np.asarray(quantum_scores)

    sorted_indices = np.argsort(quantum_scores)[::-1]
    top_indices = sorted_indices[: min(top_k, len(sorted_indices))]

    top_labels = labels_array[top_indices]
    top_classical = classical_scores[top_indices]
    top_quantum = quantum_scores[top_indices]

    y = np.arange(len(top_labels))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        y - bar_height / 2,
        top_classical,
        bar_height,
        label="Classical PageRank",
    )
    ax.barh(
        y + bar_height / 2,
        top_quantum,
        bar_height,
        label="Quantum Walk (Max Peak)",
    )

    ax.set_xlabel("Importance / Probability Score", fontsize=12)
    ax.set_title(
        f"Top {len(top_indices)} Bottleneck Nodes: PageRank vs CTQW",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(top_labels, fontsize=10)
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def write_candidate_ranking(
    output_path: Path,
    candidate_labels: list[str],
    ctqw_scores: np.ndarray,
) -> None:
    """Write the CTQW ranking used as the Stage-1 output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked_indices = np.argsort(ctqw_scores)[::-1]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["method", "rank", "node"])

        for rank, index in enumerate(ranked_indices, start=1):
            writer.writerow(
                [
                    "phase_ctqw_risk_heuristic",
                    rank,
                    candidate_labels[index],
                ]
            )

    print(f"\nStage-1 ranking saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reduced CTQW hardware/simulator demo for software "
            "supply-chain dependency analysis."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Dependency JSON path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--infected",
        default=DEFAULT_INFECTED,
        help=f"Known infected package (default: {DEFAULT_INFECTED})",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help=(
            "Maximum reachable packages included in addition to the "
            "known infected source."
        ),
    )
    parser.add_argument(
        "--t-obs",
        type=float,
        default=DEFAULT_T_OBS,
        help="CTQW observation/evolution time used for the Qiskit circuit.",
    )
    parser.add_argument(
        "--trotter-reps",
        type=int,
        default=DEFAULT_TROTTER_REPS,
        help="Lie-Trotter repetition count.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="Measurement shots.",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help=(
            "Submit to IBM Quantum hardware. By default the script "
            "uses AerSimulator."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of nodes shown in the PageRank/CTQW comparison plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Stage-1 CSV output path (default: {DEFAULT_OUTPUT_CSV})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.candidate_limit < 1:
        raise ValueError("--candidate-limit must be at least 1.")
    if args.trotter_reps < 1:
        raise ValueError("--trotter-reps must be at least 1.")
    if args.shots < 1:
        raise ValueError("--shots must be at least 1.")

    (
        padded_adjacency,
        hamiltonian,
        initial_state,
        node_labels,
        num_original_nodes,
    ) = load_dependency_subgraph(
        args.data,
        args.infected,
        args.candidate_limit,
    )

    num_qubits = int(np.log2(len(initial_state)))

    circuit = build_ctqw_circuit(
        hamiltonian,
        initial_state,
        num_qubits,
        args.t_obs,
        args.trotter_reps,
    )
    counts, backend_name = run_ctqw_circuit(
        circuit,
        num_qubits,
        args.shots,
        args.hardware,
    )
    print_measurement_probabilities(
        counts,
        num_qubits,
        args.shots,
        backend_name,
    )

    print("\n[Qiskit] Circuit")
    print(circuit.draw(output="text"))

    # Preserve the original directed propagation adjacency for PageRank.
    # Dummy states are removed before ranking is exported.
    pagerank_scores = calculate_pagerank(padded_adjacency)
    ctqw_scores = calculate_ctqw_max_peaks(
        hamiltonian,
        initial_state,
        DEFAULT_CTQW_TIME_START,
        DEFAULT_CTQW_TIME_END,
        DEFAULT_CTQW_TIME_STEPS,
    )

    plot_comparison(
        node_labels[:num_original_nodes],
        pagerank_scores[:num_original_nodes],
        ctqw_scores[:num_original_nodes],
        args.top_k,
    )

    for index in range(num_original_nodes):
        print(
            f"Node {index} ({node_labels[index]}): "
            f"{ctqw_scores[index]:.8f}"
        )

    write_candidate_ranking(
        args.output,
        node_labels[:num_original_nodes],
        ctqw_scores[:num_original_nodes],
    )


if __name__ == "__main__":
    main()
