for config in experiments/bert_*.yaml; do
  echo "Running experiment with config: $config"
  python run_experiment.py --run --config "$config"
done