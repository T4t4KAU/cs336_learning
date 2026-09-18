"""Run local experiment grids and summarize their logged validation losses."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from cs336_basics.train import parser


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True, help="config.json emitted by a training run")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["learning-rate", "batch-size", "ablation"], required=True)
    p.add_argument("--values", nargs="+", type=float)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    base = json.loads(Path(args.base_config).read_text())
    base.update(resume=None, eval_only=False)
    if args.mode == "learning-rate":
        values = args.values or [1e-4, 3e-4, 1e-3]
        ratio = base["min_lr"] / base["lr"] if base["lr"] else 0.1
        variants = [(f"lr_{lr:g}", {"lr": lr, "min_lr": lr * ratio}) for lr in values]
    elif args.mode == "batch-size":
        values = args.values or [8, 16, 32]
        if any(value <= 0 or not value.is_integer() for value in map(float, values)):
            raise ValueError("batch sizes must be positive integers")
        # Keep approximately the same total number of training tokens.
        variants = []
        for value in values:
            batch = int(value)
            ratio = base["batch_size"] / batch
            steps = max(1, round(base["steps"] * ratio))
            variants.append(
                (
                    f"batch_{batch}",
                    {
                        "batch_size": batch,
                        "steps": steps,
                        "schedule_steps": steps,
                        "warmup_steps": min(steps - 1, round(base["warmup_steps"] * ratio)),
                    },
                )
            )
    else:
        base.update(norm_style="pre", no_rope=False, ffn_type="swiglu", d_ff=None)
        variants = [
            ("baseline", {}),
            ("no_norm", {"norm_style": "none"}),
            ("post_norm", {"norm_style": "post"}),
            ("no_rope", {"no_rope": True}),
            ("silu", {"ffn_type": "silu"}),
        ]
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("experiment output directory must be empty")
    results = []
    for name, override in variants:
        config = {**base, **override, "out_dir": str(root / name)}
        cmd = [sys.executable, "-m", "cs336_basics.train"]
        for key, value in config.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            elif value is not None:
                cmd.extend([flag, str(value)])
        parser().parse_args(cmd[3:])
        print(json.dumps(cmd), flush=True)
        if args.dry_run:
            continue
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        metrics_path = root / name / "metrics.jsonl"
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        validation = [row for row in metrics if "val_loss" in row]
        results.append(
            {
                "name": name,
                "exit_code": result.returncode,
                "best_val_loss": min((r["val_loss"] for r in validation), default=None),
                "final_val_loss": validation[-1]["val_loss"] if validation else None,
            }
        )
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    if any(result["exit_code"] != 0 for result in results):
        raise SystemExit("Some runs failed; inspect summary.csv and the corresponding logs")


if __name__ == "__main__":
    main()
