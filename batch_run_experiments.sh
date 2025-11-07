#!/bin/bash

# This script runs all experiments defined in YAML files within a specified directory.
# Usage: ./run_experiment_batch.sh <directory>
# Example: ./run_experiment_batch.sh experiments/query_expansions

# Check if directory argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 experiments/query_expansions"
    echo "         $0 experiments/baselines"
    exit 1
fi

# Get the directory from command line argument and remove trailing slash if present
EXPERIMENT_DIR="${1%/}"

# Check if directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Directory '$EXPERIMENT_DIR' does not exist"
    exit 1
fi

# For each yaml file in the specified directory
for yaml_file in "$EXPERIMENT_DIR"/*.yaml; do
    # Check if any yaml files exist
    if [ ! -e "$yaml_file" ]; then
        echo "No .yaml files found in $EXPERIMENT_DIR"
        exit 1
    fi
    
    echo "Running experiment: $yaml_file"
    python experiment.py --run "$yaml_file"
done

echo "All experiments in $EXPERIMENT_DIR completed!"