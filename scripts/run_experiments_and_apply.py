"""Automate multiple DVC experiments, compare validation metrics, and apply the best model."""

import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENT_CONFIGS = [
    {
        "name": "exp_seed_42_a1",
        "params": [
            "configs/model.yaml:tuning.random_state=42",
            "configs/model.yaml:ridge.alpha=1.0",
        ],
        "desc": "Seed 42, Ridge alpha=1.0",
    },
    {
        "name": "exp_seed_101_a3",
        "params": [
            "configs/model.yaml:tuning.random_state=101",
            "configs/model.yaml:ridge.alpha=3.0",
        ],
        "desc": "Seed 101, Ridge alpha=3.0",
    },
    {
        "name": "exp_seed_777_a05",
        "params": [
            "configs/model.yaml:tuning.random_state=777",
            "configs/model.yaml:ridge.alpha=0.5",
        ],
        "desc": "Seed 777, Ridge alpha=0.5",
    },
    {
        "name": "exp_seed_999_a5",
        "params": [
            "configs/model.yaml:tuning.random_state=999",
            "configs/model.yaml:ridge.alpha=5.0",
        ],
        "desc": "Seed 999, Ridge alpha=5.0",
    },
]


def run_command(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    """Execute subprocess command and stream/capture output."""
    print(f"\n[EXEC] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    output_lines = []
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
        output_lines.append(line)
    proc.stdout.close()
    proc.wait()
    return proc.returncode, "".join(output_lines)


def get_latest_metrics() -> dict:
    """Read latest_training_summary.json to retrieve the best candidate metrics."""
    summary_path = Path("reports/evaluation/latest_training_summary.json")
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    print("=" * 70)
    print("BikeSense AI — Multi-Parameter DVC Experiments & Auto-Apply")
    print("=" * 70)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    # Resolve python and dvc executables in virtualenv
    venv_dir = Path(".venv/Scripts")
    dvc_exe = str(venv_dir / "dvc.exe") if (venv_dir / "dvc.exe").exists() else "dvc"
    python_exe = str(venv_dir / "python.exe") if (venv_dir / "python.exe").exists() else sys.executable

    results = []

    for idx, cfg in enumerate(EXPERIMENT_CONFIGS, start=1):
        name = cfg["name"]
        print(f"\n>>> Running Experiment {idx}/{len(EXPERIMENT_CONFIGS)}: '{name}' ({cfg['desc']})")
        cmd = [dvc_exe, "exp", "run", "--force", "-n", name]
        for p in cfg["params"]:
            cmd.extend(["-S", p])

        retcode, out = run_command(cmd, env)
        if retcode != 0:
            print(f"[WARN] Experiment '{name}' returned non-zero exit code: {retcode}")

        summary = get_latest_metrics()
        best_cand = summary.get("best_candidate", {})
        val_mae = best_cand.get("val_mae", float("inf"))
        val_shortfall = best_cand.get("val_shortfall", float("inf"))
        model_name = best_cand.get("model_name", "unknown")
        run_id = best_cand.get("run_id", "unknown")

        results.append(
            {
                "experiment_name": name,
                "desc": cfg["desc"],
                "best_model": model_name,
                "val_mae": val_mae,
                "val_shortfall": val_shortfall,
                "run_id": run_id,
                "exit_code": retcode,
            }
        )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<20} | {'Model':<18} | {'Val MAE':<9} | {'Shortfall':<10} | {'Status'}")
    print("-" * 70)
    for r in results:
        status = "Success" if r["exit_code"] == 0 else f"Fail({r['exit_code']})"
        print(
            f"{r['experiment_name']:<20} | {r['best_model']:<18} | {r['val_mae']:<9.4f} | {r['val_shortfall']:<10.2f} | {status}"
        )

    # Pick the winning experiment with the lowest validation MAE
    valid_results = [r for r in results if r["val_mae"] < float("inf")]
    if not valid_results:
        print("[ERROR] No successful experiments to apply.")
        sys.exit(1)

    winner = min(valid_results, key=lambda r: r["val_mae"])
    print("\n" + "=" * 70)
    print(f"WINNING EXPERIMENT: '{winner['experiment_name']}'")
    print(f"Model: {winner['best_model']} | Val MAE: {winner['val_mae']:.4f} | Shortfall: {winner['val_shortfall']:.2f}")
    print("=" * 70)

    # Automatically apply the winning experiment to the workspace
    print(f"\n[AUTO-APPLY] Applying winning experiment '{winner['experiment_name']}' to workspace...")
    apply_cmd = [dvc_exe, "exp", "apply", winner["experiment_name"]]
    retcode, _ = run_command(apply_cmd, env)
    if retcode == 0:
        print(f"[SUCCESS] Experiment '{winner['experiment_name']}' applied to workspace!")
    else:
        print(f"[NOTE] dvc exp apply finished with status {retcode}.")

    # Register in MLflow Registry as candidate
    print("\n[REGISTRY] Registering best model into MLflow Model Registry...")
    reg_cmd = [python_exe, "-m", "src.registry.register"]
    run_command(reg_cmd, env)

    # Run promotion gate evaluation
    print("\n[GATE] Evaluating promotion gate for challenger vs champion...")
    promote_cmd = [python_exe, "-m", "src.registry.promote", "--auto-approve"]
    run_command(promote_cmd, env)

    print("\n" + "=" * 70)
    print("All tasks completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
