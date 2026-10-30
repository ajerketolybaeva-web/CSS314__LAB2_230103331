
import hashlib
import json
import os


def get_student_params(student_id: str):
    """
    Generate deterministic parameters based on Student ID.
    Uses SHA-256 hashing.
    """

    h = int(
        hashlib.sha256(student_id.encode("utf-8")).hexdigest()[:8],
        16
    )

    # Salted matrix dimension
    N = 1024 + ((h % 9) * 64)

    # Buffer size
    buffer_elements = 16_777_216

    # Deterministic random seed
    seed = h % 100_007

    return {
        "student_id": student_id,
        "N": N,
        "buffer_elements": buffer_elements,
        "seed": seed
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CS482 Student ID Salting"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Your Student ID"
    )

    args = parser.parse_args()

    if not args.student_id.isdigit() or len(args.student_id) != 9:
        parser.error("Student ID must contain exactly 9 digits.")

    params = get_student_params(args.student_id)

    # Find project root directory
    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    results_dir = os.path.join(project_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(
        results_dir,
        "salted_params.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(params, file, indent=4)

    print(json.dumps(params, indent=4))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
