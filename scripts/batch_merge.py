import os
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def merge_step(base_dir, step, script_path):
    local_dir = os.path.join(base_dir, f'global_step_{step}', 'actor')
    target_dir = os.path.join(local_dir, 'huggingface')

    os.makedirs(target_dir, exist_ok=True)

    cmd = [
        "python", script_path, "merge",
        "--backend", "fsdp",
        "--tie-word-embedding",
        "--local_dir", local_dir,
        "--target_dir", target_dir
    ]

    try:
        print(f"Starting merge for model {local_dir} at step {step}...")
        subprocess.run(cmd, check=True)
        print(f"Step {step} merged successfully.")
        return step, True, None
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while merging step {step}: {e}")
        return step, False, str(e)

def merge_steps_parallel(base_dir, steps, script_path, max_workers=4):
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(merge_step, base_dir, step, script_path): step for step in steps}
        for future in as_completed(futures):
            step = futures[future]
            try:
                step, success, error = future.result()
                results.append((step, success, error))
            except Exception as exc:
                print(f"Step {step} generated an exception: {exc}")
                results.append((step, False, str(exc)))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch merge legacy models by steps with multiprocessing.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the results")
    parser.add_argument("--steps", type=int, nargs='+', required=True, help="List of step numbers to merge, e.g. --steps 60 70 80")
    parser.add_argument(
        "--script_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "legacy_model_merger.py"),
        help="Path to legacy_model_merger.py script",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="Max number of parallel processes")
    
    args = parser.parse_args()
    
    results = merge_steps_parallel(args.base_dir, args.steps, args.script_path, args.max_workers)

    print("\nSummary:")
    for step, success, error in results:
        if success:
            print(f"Step {step}: Success")
        else:
            print(f"Step {step}: Failed - {error}")