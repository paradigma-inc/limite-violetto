"""Run a fixed sequence of isolated review arms on one allocated GPU.

The JSON configuration is a nonempty list of belluxx_benchmark argument lists.
Every arm is checked with the real parser before any engine is launched.
Each subprocess owns its engine, output directory, and log. This module does
not select a scheduler or allocate hardware.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from limite_evals.belluxx_benchmark import digest, write
from limite_evals.belluxx_benchmark import parser as review_parser


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    commands = json.loads(args.config.read_text())
    if not isinstance(commands, list) or not commands:
        raise ValueError("Expected a nonempty list of argument lists")
    outputs = []
    for command in commands:
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise ValueError("Each arm must be an argument list")
        config = review_parser().parse_args(command)
        if config.phase != "run" or config.check:
            raise ValueError("Sequences contain run arms only, without --check")
        outputs.append((config.output_root or config.root / "runs") / config.run_id)
        subprocess.run(
            [sys.executable, "-m", "limite_evals.belluxx_benchmark", *command, "--check"], check=True
        )
    if len(set(outputs)) != len(outputs):
        raise ValueError("Sequence outputs must be unique")
    if args.check:
        print(f"Validated {len(commands)} isolated arms; no models or datasets loaded.")
        return
    for command, output in zip(commands, outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        # Logs sit beside the run directory so its exclusive creation still
        # protects against overwrites inside the engine process.
        with output.with_suffix(".log").open("x") as log:
            started = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-u", "-m", "limite_evals.belluxx_benchmark", *command],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        receipt = {
            "command": command,
            "returncode": result.returncode,
            "process_elapsed_s": time.perf_counter() - started,
            "sequence_config_sha256": digest(args.config),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write(output.with_suffix(".process.json"), receipt)
        result.check_returncode()


if __name__ == "__main__":
    main()
