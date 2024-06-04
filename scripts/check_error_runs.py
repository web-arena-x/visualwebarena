"""Some executions may failed.
This script checks the recordings, print the task ids.
It deletes the recordings if needed."""

import argparse
import glob
import json
import os
import sys


def merge_logs(
    args: argparse.Namespace,
    save_file: str = "tmp_merged_log.txt",
) -> tuple[str, dict[str, list[str]]]:
    if not os.path.exists(f"{args.result_folder}/log_files.txt"):
        sys.exit(1)

    with open(f"{args.result_folder}/log_files.txt", "r") as f:
        log_files = f.readlines()

    merged_results = {}
    for file in log_files:
        with open(file.strip(), "r") as f:
            lines = f.readlines()

        cur_log: list[str] = []
        index = None
        for line in lines:
            if "[Config file]" in line:
                if (
                    cur_log
                    and index
                    and os.path.exists(f"{args.result_folder}/render_{index}.html")
                    and len(cur_log) >= 3
                ):
                    merged_results[index] = cur_log
                # update index and log
                index = line.split("/")[-1].split(".")[0]
                cur_log = [line]
            else:
                cur_log.append(line)

        if (
            cur_log
            and index
            and os.path.exists(f"{args.result_folder}/render_{index}.html")
            and len(cur_log) >= 3
        ):

            merged_results[index] = cur_log

    # sort by the key
    merged_results = dict(sorted(merged_results.items(), key=lambda x: int(x[0])))

    merged_log_path = f"{args.result_folder}/{save_file}"
    with open(merged_log_path, "w") as f:
        for k, v in merged_results.items():
            for line in v:
                f.write(line)
    print(f"Number of examples: {len(merged_results)}")

    return merged_log_path, merged_results


def merge_eval_cache(
    args: argparse.Namespace,
    save_file: str = "tmp_merged_eval_cache.jsonl"
) -> None:
    if not os.path.exists(f"{args.result_folder}/log_files.txt"):
        sys.exit(1)

    with open(f"{args.result_folder}/log_files.txt", "r") as f:
        log_files = f.readlines()

    id_to_cache = {}
    for file in log_files:
        with open(file.strip().replace(".log", "_eval_cache.jsonl"), "r") as f:
            for line in f:
                data = json.loads(line)
                id = data["config_file"].split(".")[0]
                # later will overwrite the previous data
                id_to_cache[id] = data

    merged_cache_path = f"{args.result_folder}/{save_file}"
    # sort by the key
    id_to_cache = dict(sorted(id_to_cache.items(), key=lambda x: int(x[0])))
    with open(merged_cache_path, "w") as f:
        for k, v in id_to_cache.items():
            f.write(json.dumps(v) + "\n")


def check_unlogged(
    args: argparse.Namespace,
    merged_results: dict[str, list],
) -> None:
    unlog_examples = []
    for i in range(812):
        if (
            os.path.exists(f"{args.result_folder}/render_{i}.html")
            and str(i) not in merged_results
        ):
            unlog_examples.append(i)

    print(f"Number of unlogged examples: {len(unlog_examples)}")
    print(unlog_examples)
    if (
        args.delete_errors
        or input("Do you want to delete these examples? (y/n)") == "y"
    ):
        for idx in unlog_examples:
            os.remove(f"{args.result_folder}/render_{idx}.html")


def check_unfinished(merged_results: dict[str, list]) -> None:
    unifinished_examples = [i for i in range(0, 812) if str(i) not in merged_results]
    print(f"Number of unfinished examples: {len(unifinished_examples)}")
    print(unifinished_examples)


def check_unhandled_errors(args: argparse.Namespace, log_path: str) -> int:
    with open(log_path, "r") as f:
        logs = f.read()

    error_examples = []
    for line in logs.split("\n"):
        if "[Config file]" in line:
            example_idx = line.split("/")[-1].split(".")[0]
        if "[Unhandled Error]" in line or "[OpenAI Error]" in line:
            error_examples.append(int(example_idx))

    num_errors = len(error_examples)
    print(f"Number of unhandled errors: {len(error_examples)}")
    print(error_examples)
    if (
        args.delete_errors
        or input("Do you want to delete these examples? (y/n)") == "y"
    ):
        for idx in error_examples:
            if os.path.exists(f"{args.result_folder}/render_{idx}.html"):
                os.remove(f"{args.result_folder}/render_{idx}.html")
    return num_errors


def check_unexpected_logout(args: argparse.Namespace) -> int:
    target_strings = set(
        [
            "Creating an account has many benefits: check out faster",
            "Welcome, please sign in",
            "Username or email",
            "Keep me logged in",
        ]
    )

    error_examples = []
    for render_file in glob.glob(f"{args.result_folder}/render_*.html"):
        with open(render_file, "r") as f:
            contents = f.read()
            if any([s in contents for s in target_strings]):
                task_id = int(render_file.split("/")[-1].split(".")[0].split("_")[-1])
                error_examples.append(task_id)
    print(f"Number of unexpected logout: {len(error_examples)}")
    print(error_examples)
    num_errors = len(error_examples)
    if (
        args.delete_errors
        or input("Do you want to delete these examples? (y/n)") == "y"
    ):
        for idx in error_examples:
            if os.path.exists(f"{args.result_folder}/render_{idx}.html"):
                os.remove(f"{args.result_folder}/render_{idx}.html")

    return num_errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_folder", type=str)
    parser.add_argument("--delete_errors", action="store_true")
    parser.add_argument("--tolerance", type=int, default=0)

    args = parser.parse_args()
    log_path, merged_results = merge_logs(args)
    merge_eval_cache(args)
    check_unlogged(args, merged_results)
    check_unfinished(merged_results)
    n1 = check_unhandled_errors(args, log_path)
    n2 = check_unexpected_logout(args)
    if n1 + n2 > args.tolerance:
        sys.exit(1)
    else:
        sys.exit(0)
