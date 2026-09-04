#!/bin/bash

for config in experiments/new_runs/*.yaml; do
    echo "Running experiment: $config"
    python experiment.py --run "$config"
done

echo "All experiments complete."
