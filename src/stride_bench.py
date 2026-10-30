
import os
import csv
import time
import numpy as np


# Benchmark configuration
BUFFER_ELEMENTS = 16_777_216

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]

WARMUP_RUNS = 5
RECORDED_RUNS = 5


def benchmark_stride(data, stride):
    """
    Measure memory reading performance for a given stride.
    """

    total = 0.0
    start_time = time.perf_counter_ns()

    for i in range(0, len(data), stride):
        total += data[i]

    end_time = time.perf_counter_ns()

    elapsed_ns = end_time - start_time

    elements_touched = len(range(0, len(data), stride))

    latency_per_element = elapsed_ns / elements_touched

    return {
        "stride": stride,
        "elapsed_ns": elapsed_ns,
        "elements_touched": elements_touched,
        "latency_ns_per_element": latency_per_element,
        "checksum": total
    }


def run_benchmark():
    print(f"Allocating {BUFFER_ELEMENTS:,} elements...")

    data = np.ones(BUFFER_ELEMENTS, dtype=np.float64)

    results = []

    for stride in STRIDES:

        print(f"\nTesting stride: {stride}")

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            benchmark_stride(data, stride)

        # Recorded runs
        recorded_times = []
        last_result = None

        for run in range(RECORDED_RUNS):
            result = benchmark_stride(data, stride)

            recorded_times.append(result["elapsed_ns"])
            last_result = result

            print(
                f"Run {run + 1}: "
                f"{result['elapsed_ns'] / 1_000_000:.3f} ms"
            )

        # Calculate median elapsed time
        median_time_ns = float(np.median(recorded_times))

        elements_touched = last_result["elements_touched"]

        latency = median_time_ns / elements_touched

        results.append({
        "stride": stride,
        "median_time_ns": median_time_ns,
        "elements_touched": elements_touched,
        "latency_ns": latency,
        "latency_ns_per_element": latency,
        "checksum": last_result["checksum"]
})

        print(f"Median: {median_time_ns / 1_000_000:.3f} ms")
        print(f"Latency: {latency:.4f} ns/element")

    return results


def save_results(results):
    """
    Save benchmark results to TSV.
    """

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    results_dir = os.path.join(project_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(
        results_dir,
        "stride_results.tsv"
    )

    with open(output_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
             "stride",
            "median_time_ns",
            "elements_touched",
            "latency_ns",
            "latency_ns_per_element",
            "checksum"
            ],
            delimiter="\t"
        )

        writer.writeheader()
        writer.writerows(results)

    print(output_path)


def main():
    results = run_benchmark()

    save_results(results)

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
