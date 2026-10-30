
import os
import json
import time
import numpy as np
from numba import njit


def load_parameters():
    """Load salted parameters from JSON."""

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    params_path = os.path.join(
        project_dir, "results", "salted_params.json"
    )

    with open(params_path, "r", encoding="utf-8") as file:
        params = json.load(file)

    return params["N"], params["seed"]


@njit
def multiply_ijk(A, B):
    """Naive matrix multiplication: i-j-k."""

    M = A.shape[0]
    C = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        for j in range(M):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]

    return C


@njit
def multiply_ikj(A, B):
    """Interchanged matrix multiplication: i-k-j."""

    M = A.shape[0]
    C = np.zeros((M, M), dtype=np.float64)

    for i in range(M):
        for k in range(M):
            a = A[i, k]

            for j in range(M):
                C[i, j] += a * B[k, j]

    return C


def measure_runtime(function, A, B, runs=3):
    """Measure median runtime across multiple runs."""

    times = []
    result = None

    # Warmup: compile Numba function
    function(A, B)

    for run in range(runs):
        start = time.perf_counter()

        result = function(A, B)

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"Run {run + 1}: {elapsed:.4f} seconds")

    median_time = float(np.median(times))

    return median_time, result


def main():
    # Load student parameters
    N, seed = load_parameters()

    M = N // 2

    print(f"Student salted N: {N}")
    print(f"Matrix size M: {M}")
    print(f"Random seed: {seed}")

    # Initialize matrices
    rng = np.random.default_rng(seed)

    A = rng.random((M, M), dtype=np.float64)
    B = rng.random((M, M), dtype=np.float64)

    print("\nMatrices initialized.")

    # Algorithm A: i-j-k
    print("\nTesting Algorithm A: i-j-k")

    time_ijk, C_ijk = measure_runtime(
        multiply_ijk, A, B, runs=3
    )

    # Algorithm B: i-k-j
    print("\nTesting Algorithm B: i-k-j")

    time_ikj, C_ikj = measure_runtime(
        multiply_ikj, A, B, runs=3
    )

    # Calculate speedup
    speedup = time_ijk / time_ikj

    # Numerical verification
    max_difference = float(np.max(np.abs(C_ijk - C_ikj)))

    is_correct = np.allclose(
        C_ijk,
        C_ikj,
        rtol=1e-12,
        atol=1e-12
    )

    # Results
    results = {
    "student_id": "230103224",
    "N": N,
    "M": M,

    # Fields required by teacher's verifier
    "naive_time_s": time_ijk,
    "interchanged_time_s": time_ikj,
    "speedup": speedup,

    # Original experiment data
    "algorithm_ijk_median_seconds": time_ijk,
    "algorithm_ikj_median_seconds": time_ikj,
    "max_absolute_difference": max_difference,
    "numerical_verification": bool(is_correct)
}

    # Save JSON
    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(
        results_dir, "loop_interchange.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"i-j-k median: {time_ijk:.4f} seconds")
    print(f"i-k-j median: {time_ikj:.4f} seconds")
    print(f"Speedup: {speedup:.4f}x")
    print(f"Max difference: {max_difference:.12e}")
    print(f"Verification: {'PASSED' if is_correct else 'FAILED'}")

    print("\nResults saved:", output_path)


if __name__ == "__main__":
    main()
