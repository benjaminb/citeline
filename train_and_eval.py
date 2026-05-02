"""
Unified train-then-evaluate script.

Reads a single YAML config with top-level 'training' and 'experiment' sections.
The output directory is derived from the config filename: results are written to
experiments/train_and_eval/results/{config_stem}/{version}/, where version is
auto-incremented each run (1, 2, 3, ...).

Usage:
    python train_and_eval.py --config path/to/unified_config.yaml
"""

import argparse
import os
import shutil
import sys
import yaml
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train an adapter then evaluate via experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to unified YAML config.")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main():
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config) as f:
        unified = yaml.safe_load(f)

    config_name = Path(args.config).stem
    base_results_dir = Path("experiments/train_and_eval/results") / config_name
    version = 1
    while (base_results_dir / str(version)).exists():
        version += 1
    output_dir = base_results_dir / str(version)
    print(f"Run version: {version}  →  {output_dir}")

    training_cfg = unified.get("training", {})
    experiment_cfg = unified.get("experiment", {})

    # Create output directory structure
    model_dir = output_dir / "model"
    experiment_dir = output_dir / "experiment"
    model_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Copy unified config for reproducibility
    shutil.copy(args.config, output_dir / "config.yaml")
    print(f"Config saved to: {output_dir / 'config.yaml'}")

    # --- Training ---
    print("\n" + "=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)

    from citeline.nn.config import TrainConfig
    from citeline.nn.train import run_training

    train_config = TrainConfig(**training_cfg)
    best_model_path = run_training(train_config, checkpoint_dir=str(model_dir))

    if not best_model_path.exists():
        print(f"ERROR: Expected best model at {best_model_path} but file not found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining complete. Best model: {best_model_path}")

    # --- Experiment ---
    print("\n" + "=" * 60)
    print("PHASE 2: EXPERIMENT")
    print("=" * 60)

    assert "CPUS" in os.environ, "CPUS environment variable must be set for the experiment phase."

    from experiment import Experiment

    experiment_cfg["nn"] = str(best_model_path)
    experiment_cfg["output_path"] = str(experiment_dir) + "/"

    experiment = Experiment(**experiment_cfg)
    experiment.run()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Config:           {output_dir / 'config.yaml'}")
    print(f"  Loss plot:        {model_dir / 'loss_history.png'}")
    print(f"  Margin plot:      {model_dir / 'margin_history.png'}")
    print(f"  Similarity plot:  {model_dir / 'similarity_history.png'}")
    print(f"  Model:            {model_dir}/")
    print(f"  Results:          {experiment_dir}/")


if __name__ == "__main__":
    main()
