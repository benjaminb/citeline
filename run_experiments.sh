#!/bin/bash

# Files that experiment.py writes into output_path on a completed run.
OUTPUT_FILES=("results.json" "ndcg_chunks_at_k.npy" "ndcg_docs_at_k.npy")

# Set FORCE=1 to re-run experiments even if their outputs are already present.
FORCE="${FORCE:-0}"

for config in experiments/new_runs/*.yaml; do
    # Pull output_path out of the config, stripping optional quotes and comments.
    output_path=$(sed -n 's/^output_path:[[:space:]]*//p' "$config" | head -n 1 | sed "s/[[:space:]]*#.*//; s/^['\"]//; s/['\"]$//")

    if [ "$FORCE" != "1" ] && [ -n "$output_path" ]; then
        complete=1
        for f in "${OUTPUT_FILES[@]}"; do
            [ -f "$output_path/$f" ] || complete=0
        done
        if [ "$complete" = "1" ]; then
            echo "Skipping experiment (outputs exist in $output_path): $config"
            continue
        fi
    fi

    echo "Running experiment: $config"
    python experiment.py --run "$config"
done

echo "All experiments complete."
