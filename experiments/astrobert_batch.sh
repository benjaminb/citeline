for config in experiments/astrobert_*.yaml; do
  echo "Running experiment with config: $config"
  python run_experiment.py --run --config "$config"
done