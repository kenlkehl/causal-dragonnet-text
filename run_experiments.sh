#!/bin/bash
set -e

EXPERIMENTS_DIR="../pcori_experiments/rethink_confounders_real_data/experiments_1-5-26"

# Find all config files
CONFIGS=$(find "$EXPERIMENTS_DIR" -name "config.json" | sort)

for config in $CONFIGS; do
    echo "================================================================================"
    echo "Running experiment with config: $config"
    echo "================================================================================"
    
    # Run the experiment
    ~/pcori/bin/cdt run --config "$config"
    
    echo "Experiment finished."
    echo ""
done

echo "All experiments completed successfully."
