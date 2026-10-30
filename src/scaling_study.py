
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import csv
import json
import os
import math
import platform
import statistics
import atexit


# ============================================================
# CONFIGURATION
# ============================================================

BASE_N = 2048

PROCESS_COUNTS = [1, 2, 4, 8, 11]

REPEATS = 5

DTYPE = np.float64
DTYPE_NAME = "float64"

FLOPS_PER_CELL = 5

# Naive model: 5 reads + 1 write.
NAIVE_BYTES_PER_CELL = 6 * 8

# Ideal cache reuse: one input read + one output write.
IDEAL_BYTES_PER_CELL = 2 * 8

MEMORY_BANDWIDTH_GB_S = 150.0

RESULTS_DIR = "results"

SCALING_FILE = os.path.join(
    RESULTS_DIR,
    "scaling_metrics.tsv"
)

ROOFLINE_FILE = os.path.join(
    RESULTS_DIR,
    "roofline_summary.json"
)


# ============================================================
# SHARED WORKER MEMORY
# ============================================================

_SOURCE = None
_OUTPUT = None
_INPUT_SHM = None
_OUTPUT_SHM = None


def close_worker_memory():

    global _INPUT_SHM, _OUTPUT_SHM

    if _INPUT_SHM is not None:
        _INPUT_SHM.close()

    if _OUTPUT_SHM is not None:
        _OUTPUT_SHM.close()


def initialize_worker(input_name, output_name, n):

    global _SOURCE
    global _OUTPUT
    global _INPUT_SHM
    global _OUTPUT_SHM

    _INPUT_SHM = shared_memory.SharedMemory(
        name=input_name
    )

    _OUTPUT_SHM = shared_memory.SharedMemory(
        name=output_name
    )

    _SOURCE = np.ndarray(
        (n, n),
        dtype=DTYPE,
        buffer=_INPUT_SHM.buf
    )

    _OUTPUT = np.ndarray(
        (n, n),
        dtype=DTYPE,
        buffer=_OUTPUT_SHM.buf
    )

    atexit.register(close_worker_memory)


# ============================================================
# STENCIL KERNEL
# ============================================================

def stencil_worker(bounds):

    start_row, end_row = bounds

    _OUTPUT[start_row:end_row, 1:-1] = (
        _SOURCE[start_row:end_row, 1:-1]
        + _SOURCE[start_row-1:end_row-1, 1:-1]
        + _SOURCE[start_row+1:end_row+1, 1:-1]
        + _SOURCE[start_row:end_row, :-2]
        + _SOURCE[start_row:end_row, 2:]
    ) * 0.2


# ============================================================
# DOMAIN DECOMPOSITION
# ============================================================

def get_chunks(n, processes):

    interior_rows = n - 2

    chunks = []

    for worker_id in range(processes):

        start = (
            1 + worker_id * interior_rows // processes
        )

        end = (
            1 + (worker_id + 1) * interior_rows // processes
        )

        if start < end:
            chunks.append((start, end))

    return chunks


# ============================================================
# INITIALIZE MATRIX
# ============================================================

def initialize_matrix(source, output, n):

    rng = np.random.default_rng(34301)

    # Fill row by row to avoid allocating another
    # full-size temporary matrix.

    for row in range(n):
        source[row, :] = rng.random(n)

    output.fill(0)


# ============================================================
# VALIDATION
# ============================================================

def validate_result(source, output, n):

    result = output[1:-1, 1:-1]

    if not np.isfinite(result).all():
        raise RuntimeError("Invalid stencil output")

    # Validate selected cells against the mathematical formula.

    test_points = [
        (1, 1),
        (n // 4, n // 4),
        (n // 2, n // 2),
        (n - 2, n - 2)
    ]

    for i, j in test_points:

        expected = (
            source[i, j]
            + source[i - 1, j]
            + source[i + 1, j]
            + source[i, j - 1]
            + source[i, j + 1]
        ) * 0.2

        if not np.isclose(
            output[i, j],
            expected,
            rtol=1e-12,
            atol=1e-12
        ):
            raise RuntimeError(
                f"Validation failed at ({i}, {j})"
            )

    return float(np.sum(result))


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(n, processes):

    ctx = mp.get_context("spawn")

    array_bytes = (
        n * n * np.dtype(DTYPE).itemsize
    )

    input_shm = shared_memory.SharedMemory(
        create=True,
        size=array_bytes
    )

    output_shm = shared_memory.SharedMemory(
        create=True,
        size=array_bytes
    )

    source = np.ndarray(
        (n, n),
        dtype=DTYPE,
        buffer=input_shm.buf
    )

    output = np.ndarray(
        (n, n),
        dtype=DTYPE,
        buffer=output_shm.buf
    )

    pool = None

    try:

        initialize_matrix(
            source,
            output,
            n
        )

        chunks = get_chunks(n, processes)

        pool = ctx.Pool(
            processes=processes,
            initializer=initialize_worker,
            initargs=(
                input_shm.name,
                output_shm.name,
                n
            )
        )

        # Warm-up: initialize worker processes and
        # execute one stencil before timed measurements.

        pool.map(
            stencil_worker,
            chunks
        )

        measurements = []

        for repeat in range(REPEATS):

            start = time.perf_counter()

            pool.map(
                stencil_worker,
                chunks
            )

            runtime = (
                time.perf_counter() - start
            )

            measurements.append(runtime)

            print(
                f"    Repeat {repeat + 1}/{REPEATS}: "
                f"{runtime:.6f} sec"
            )

        # Median is less sensitive to scheduling noise.

        median_runtime = statistics.median(
            measurements
        )

        checksum = validate_result(
            source,
            output,
            n
        )

        return median_runtime, checksum

    finally:

        if pool is not None:

            pool.close()
            pool.join()

        del source
        del output

        input_shm.close()
        input_shm.unlink()

        output_shm.close()
        output_shm.unlink()


# ============================================================
# AMDAHL'S LAW
# ============================================================

def calculate_amdahl_fraction(t1, tp, processes):

    if processes == 1:
        return None

    speedup = t1 / tp

    return (
        (1.0 / speedup - 1.0 / processes)
        / (1.0 - 1.0 / processes)
    )


def run_strong_scaling():

    print("\n" + "=" * 65)
    print("TASK 3.1: STRONG SCALING")
    print("=" * 65)

    results = {}

    for processes in PROCESS_COUNTS:

        print(
            f"\nStrong scaling: P={processes}, N={BASE_N}"
        )

        runtime, checksum = benchmark(
            BASE_N,
            processes
        )

        results[processes] = {
            "runtime": runtime,
            "checksum": checksum,
            "n": BASE_N
        }

        print(f"Median runtime: {runtime:.6f} sec")

    t1 = results[1]["runtime"]

    rows = []
    serial_fractions = []

    for processes in PROCESS_COUNTS:

        tp = results[processes]["runtime"]

        speedup = t1 / tp

        efficiency = (
            speedup / processes * 100.0
        )

        serial_fraction = calculate_amdahl_fraction(
            t1,
            tp,
            processes
        )

        valid_fraction = (
            serial_fraction is not None
            and 0 <= serial_fraction <= 1
        )

        if valid_fraction:
            serial_fractions.append(serial_fraction)

        rows.append({
            "Study": "strong",
            "Processes": processes,
            "Matrix_N": BASE_N,
            "Work_cells": (BASE_N - 2) ** 2,
            "Runtime_sec": tp,
            "Speedup": speedup,
            "Efficiency_percent": efficiency,
            "Serial_fraction": (
                serial_fraction
                if serial_fraction is not None
                else ""
            ),
            "Amdahl_valid": (
                valid_fraction
                if processes > 1
                else ""
            ),
            "Scaled_speedup": ""
        })

    # Do not calculate a theoretical Amdahl ceiling
    # from invalid serial fractions.

    all_valid = (
        len(serial_fractions) == len(PROCESS_COUNTS) - 1
    )

    if all_valid:

        average_s = statistics.mean(
            serial_fractions
        )

        maximum_speedup = (
            1.0 / average_s
            if average_s > 0
            else None
        )

    else:

        average_s = None
        maximum_speedup = None

    print("\nAMDAHL ANALYSIS")

    print(
        f"Valid serial fraction measurements: "
        f"{len(serial_fractions)}/{len(PROCESS_COUNTS) - 1}"
    )

    if maximum_speedup is not None:

        print(f"Average serial fraction: {average_s:.6f}")
        print(f"Maximum theoretical speedup: {maximum_speedup:.6f}")

    else:

        print(
            "Classical Amdahl fit is not valid "
            "for all measured process counts."
        )

        print(
            "Theoretical maximum speedup: N/A"
        )

    return (
        rows,
        results,
        average_s,
        maximum_speedup
    )


# ============================================================
# WEAK SCALING / GUSTAFSON
# ============================================================

def run_weak_scaling(strong_results):

    print("\n" + "=" * 65)
    print("TASK 3.2: WEAK SCALING")
    print("=" * 65)

    results = {}

    for processes in PROCESS_COUNTS:

        scaled_n = math.ceil(
            BASE_N * math.sqrt(processes)
        )

        print(
            f"\nWeak scaling: P={processes}, "
            f"N={scaled_n}"
        )

        runtime, checksum = benchmark(
            scaled_n,
            processes
        )

        work = (scaled_n - 2) ** 2

        results[processes] = {
            "runtime": runtime,
            "checksum": checksum,
            "n": scaled_n,
            "work": work
        }

        print(f"Median runtime: {runtime:.6f} sec")

    work1 = (BASE_N - 2) ** 2

    t1 = strong_results[1]["runtime"]

    rows = []

    for processes in PROCESS_COUNTS:

        tp = results[processes]["runtime"]

        work_p = results[processes]["work"]

        scaled_speedup = (
            processes * work1 * t1
            / (work_p * tp)
        )

        rows.append({
            "Study": "weak",
            "Processes": processes,
            "Matrix_N": results[processes]["n"],
            "Work_cells": work_p,
            "Runtime_sec": tp,
            "Speedup": "",
            "Efficiency_percent": "",
            "Serial_fraction": "",
            "Amdahl_valid": "",
            "Scaled_speedup": scaled_speedup
        })

        print(
            f"P={processes}: "
            f"Scaled speedup={scaled_speedup:.6f}"
        )

    return rows, results


# ============================================================
# SAVE TSV
# ============================================================

def save_scaling_results(rows):

    fieldnames = [
        "Study",
        "Processes",
        "Matrix_N",
        "Work_cells",
        "Runtime_sec",
        "Speedup",
        "Efficiency_percent",
        "Serial_fraction",
        "Amdahl_valid",
        "Scaled_speedup"
    ]

    with open(
        SCALING_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t"
        )

        writer.writeheader()

        for row in rows:

            formatted = {}

            for key, value in row.items():

                if isinstance(value, float):
                    formatted[key] = round(value, 8)
                else:
                    formatted[key] = value

            writer.writerow(formatted)

    print(f"\nSaved: {SCALING_FILE}")


# ============================================================
# ROOFLINE ANALYSIS
# ============================================================

def generate_roofline_summary(
    strong_results,
    serial_fraction,
    maximum_speedup
):

    n = BASE_N

    work_cells = (n - 2) ** 2

    total_flops = (
        work_cells * FLOPS_PER_CELL
    )

    naive_bytes = (
        work_cells * NAIVE_BYTES_PER_CELL
    )

    ideal_bytes = (
        work_cells * IDEAL_BYTES_PER_CELL
    )

    naive_ai = (
        FLOPS_PER_CELL / NAIVE_BYTES_PER_CELL
    )

    ideal_ai = (
        FLOPS_PER_CELL / IDEAL_BYTES_PER_CELL
    )

    runtime = strong_results[1]["runtime"]

    # These are modeled bandwidth estimates,
    # NOT hardware-counter DRAM measurements.

    modeled_naive_bandwidth = (
        naive_bytes / runtime / 1e9
    )

    modeled_ideal_bandwidth = (
        ideal_bytes / runtime / 1e9
    )

    summary = {
        "regime": "memory-bound",
        "arithmetic_intensity": naive_ai,
        
        "hardware": {
            "cpu": "Apple M3 Pro",
            "architecture": platform.machine(),
            "physical_cores": 11,
            "logical_cores": mp.cpu_count(),
            "peak_memory_bandwidth_gb_s": MEMORY_BANDWIDTH_GB_S,
            "cache_line_bytes": 128
        },

        "cache_hierarchy": {
            "L1D_E_core_bytes": 65536,
            "L1I_E_core_bytes": 131072,
            "L2_E_core_bytes": 4194304,
            "L1D_P_core_bytes": 131072,
            "L1I_P_core_bytes": 196608,
            "L2_P_core_bytes": 16777216,
            "L3_bytes": None,
            "note": (
                "L3 was not established by the available "
                "Milestone 0 measurements."
            )
        },

        "kernel": {
            "name": "2D five-point stencil",
            "matrix_dimension": n,
            "dtype": DTYPE_NAME,
            "flops_per_cell": FLOPS_PER_CELL,
            "work_cells": work_cells,
            "total_flops": total_flops,

            "naive_bytes_per_cell": NAIVE_BYTES_PER_CELL,
            "ideal_bytes_per_cell": IDEAL_BYTES_PER_CELL,

            "naive_estimated_bytes": naive_bytes,
            "ideal_estimated_bytes": ideal_bytes
        },

        "roofline": {

            "naive_arithmetic_intensity": naive_ai,

            "ideal_cache_reuse_arithmetic_intensity": ideal_ai,

            "peak_memory_bandwidth_gb_s": MEMORY_BANDWIDTH_GB_S,

            "modeled_naive_bandwidth_gb_s": modeled_naive_bandwidth,

            "modeled_ideal_bandwidth_gb_s": modeled_ideal_bandwidth,

            "peak_fp64_flops": None,

            "ridge_point_flops_per_byte": None,

            "classification": "Provisional: likely memory-bandwidth-sensitive",

            "reason": (
                "The stencil has low arithmetic intensity. "
                "Neighboring input values can be reused through "
                "the cache, so the naive 48-byte model and ideal "
                "16-byte model represent different traffic assumptions. "
                "Actual DRAM traffic requires hardware performance "
                "counters. Exact roofline classification requires "
                "peak FP64 compute throughput and measured DRAM bandwidth."
            )
        },

        "amdahl": {

            "effective_serial_fraction": serial_fraction,

            "maximum_speedup": maximum_speedup,

            "interpretation": (
                "Available only when all measured effective "
                "serial fractions are within [0, 1]."
            )
        }

    }

    with open(
        ROOFLINE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(f"Saved: {ROOFLINE_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("MILESTONE 3: SCALING & ROOFLINE EVALUATION")
    print("=" * 65)

    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Logical CPU count: {mp.cpu_count()}")
    print(f"Base matrix: {BASE_N} x {BASE_N}")
    print(f"Processes: {PROCESS_COUNTS}")
    print(f"Repeats: {REPEATS}")
    print("Worker pool: persistent during each benchmark")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Task 3.1

    (
        strong_rows,
        strong_results,
        serial_fraction,
        maximum_speedup
    ) = run_strong_scaling()

    # Task 3.2

    weak_rows, weak_results = run_weak_scaling(
        strong_results
    )

    # Save results

    all_rows = strong_rows + weak_rows

    save_scaling_results(all_rows)

    # Task 3.3

    generate_roofline_summary(
        strong_results,
        serial_fraction,
        maximum_speedup
    )

    print("\n" + "=" * 65)
    print("MILESTONE 3 COMPLETED")
    print("=" * 65)

    print("Task 3.1: Strong Scaling")
    print("Task 3.2: Weak Scaling")
    print("Task 3.3: Roofline Analysis")

    print("\nGenerated files:")

    print(SCALING_FILE)
    print(ROOFLINE_FILE)

    print("=" * 65)


if __name__ == "__main__":

    mp.freeze_support()
    main()
