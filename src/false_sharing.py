
import multiprocessing as mp
import ctypes
import time
import csv
import os
import platform

PROCESS_COUNTS = [1, 2, 4, 8]
ITERATIONS = 5_000_000

CACHE_LINE_SIZE = 64
INT64_SIZE = ctypes.sizeof(ctypes.c_int64)

PADDING = CACHE_LINE_SIZE // INT64_SIZE

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "false_sharing_karp_flatt.tsv"
)


def worker_unpadded(counter_array, worker_id, iterations):
    """
    Pathological implementation.

    Each process updates an adjacent int64 counter.
    Multiple counters share the same 64-byte cache line.
    """

    for _ in range(iterations):
        counter_array[worker_id] += 1


def worker_padded(counter_array, worker_id, iterations):
    """
    Padded implementation.

    Each process updates a counter separated
    by 64 bytes (8 int64 values).
    """

    index = worker_id * PADDING

    for _ in range(iterations):
        counter_array[index] += 1


def worker_privatized(counter_array, worker_id, iterations):
    """
    Thread/process-local privatization.

    Each process increments a local variable.
    Only one shared write happens after the loop.
    """

    local_counter = 0

    for _ in range(iterations):
        local_counter += 1

    counter_array[worker_id] = local_counter


def run_benchmark(implementation, worker_function, processes):
    """
    Runs one benchmark and returns execution time.
    """

    if implementation == "unpadded":
        shared_array = mp.Array(
            ctypes.c_int64,
            processes,
            lock=False
        )

    elif implementation == "padded":
        shared_array = mp.Array(
            ctypes.c_int64,
            processes * PADDING,
            lock=False
        )

    else:
        shared_array = mp.Array(
            ctypes.c_int64,
            processes,
            lock=False
        )

    process_list = []

    start_time = time.perf_counter()

    for worker_id in range(processes):

        process = mp.Process(
            target=worker_function,
            args=(
                shared_array,
                worker_id,
                ITERATIONS
            )
        )

        process_list.append(process)
        process.start()

    for process in process_list:
        process.join()

    end_time = time.perf_counter()

    runtime = end_time - start_time

    # Check process exit codes.
    for process in process_list:
        if process.exitcode != 0:
            raise RuntimeError(
                f"Worker failed: exit code {process.exitcode}"
            )

    # Validate final counters.
    if implementation == "padded":
        results = [
            shared_array[i * PADDING]
            for i in range(processes)
        ]
    else:
        results = [
            shared_array[i]
            for i in range(processes)
        ]

    expected = ITERATIONS

    if any(value != expected for value in results):
        raise RuntimeError(
            f"Incorrect result for {implementation}, "
            f"P={processes}: {results}"
        )

    return runtime



def calculate_karp_flatt(t1, tp, processes):

    if processes == 1:
        return None

    return (
        (processes * tp / t1) - 1
    ) / (processes - 1)



def main():

    print("=" * 65)
    print("MILESTONE 2: FALSE SHARING EXPERIMENT")
    print("=" * 65)

    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"CPU cores (logical): {mp.cpu_count()}")
    print(f"Iterations per process: {ITERATIONS:,}")
    print(f"Cache line assumption: {CACHE_LINE_SIZE} bytes")
    print(f"Padding: {PADDING} int64 values")

    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    implementations = {
        "unpadded": worker_unpadded,
        "padded": worker_padded,
        "privatized": worker_privatized
    }

    raw_results = {}



    for implementation, worker_function in implementations.items():

        print(f"\nIMPLEMENTATION: {implementation.upper()}")
        print("-" * 65)

        raw_results[implementation] = {}

        for processes in PROCESS_COUNTS:

            print(
                f"Running P={processes} "
                f"with {ITERATIONS:,} iterations..."
            )

            runtime = run_benchmark(
                implementation,
                worker_function,
                processes
            )

            raw_results[implementation][processes] = runtime

            print(
                f"Runtime: {runtime:.6f} seconds"
            )



    rows = []

    for implementation in implementations:

        baseline = raw_results[implementation][1]

        for processes in PROCESS_COUNTS:

            runtime = raw_results[implementation][processes]

            speedup = baseline / runtime

            efficiency = speedup / processes

            karp_flatt = calculate_karp_flatt(
                baseline,
                runtime,
                processes
            )

            rows.append({
                "Implementation": implementation,
                "Processes": processes,
                "Iterations": ITERATIONS,
                "Runtime_sec": round(runtime, 6),
                "Speedup": round(speedup, 6),
                "Efficiency": round(efficiency, 6),
                "Karp_Flatt": (
                    round(karp_flatt, 6)
                    if karp_flatt is not None
                    else ""
                )
            })



    fieldnames = [
        "Implementation",
        "Processes",
        "Iterations",
        "Runtime_sec",
        "Speedup",
        "Efficiency",
        "Karp_Flatt"
    ]

    with open(
        RESULTS_FILE,
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
        writer.writerows(rows)



    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(
        f"{'Implementation':<15}"
        f"{'P':<5}"
        f"{'Runtime':<14}"
        f"{'Speedup':<12}"
        f"{'Karp-Flatt':<12}"
    )

    print("-" * 65)

    for row in rows:

        print(
            f"{row['Implementation']:<15}"
            f"{row['Processes']:<5}"
            f"{row['Runtime_sec']:<14}"
            f"{row['Speedup']:<12}"
            f"{str(row['Karp_Flatt']):<12}"
        )

    print("=" * 65)
    print(f"Results saved to: {RESULTS_FILE}")
    print("All counter validation checks passed.")
    print("=" * 65)

if __name__ == "__main__":
    mp.freeze_support()
    main()
