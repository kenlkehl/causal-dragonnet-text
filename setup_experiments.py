import json
import os
import glob
from pathlib import Path

source_base = Path("../pcori_experiments/rethink_confounders_as_queries/experiments_1-4-26")
target_base = Path("../pcori_experiments/rethink_confounders_real_data/experiments_1-5-26")

dataset_path = "/data1/ken/pcori/analysis_dataset_phi.parquet"

# Create target base if not exists
target_base.mkdir(parents=True, exist_ok=True)

experiment_dirs = sorted([d for d in source_base.iterdir() if d.is_dir() and d.name.startswith("exp")])

keys_to_remove = ['arctanh_transform', 'aggregator_mode', 'features_per_confounder']

def clean_architecture_config(arch_config):
    if not isinstance(arch_config, dict):
        return
    for k in keys_to_remove:
        if k in arch_config:
            del arch_config[k]

for exp_dir in experiment_dirs:
    exp_name = exp_dir.name
    print(f"Processing {exp_name}...")
    
    # Read config
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"Skipping {exp_name}, no config.json found.")
        continue
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Update config
    # 1. Update paths
    target_exp_dir = target_base / exp_name
    target_exp_dir.mkdir(exist_ok=True)
    
    config['output_dir'] = str(target_exp_dir / "results")
    config['cache_dir'] = str(target_exp_dir / "cache")
    
    # 2. Update dataset path and split
    if 'applied_inference' in config:
        config['applied_inference']['dataset_path'] = dataset_path
        config['applied_inference']['split_column'] = "split" # Use the fixed split in real data
    
    # 3. Clean architecture configs
    if 'pretraining' in config and 'architecture' in config['pretraining']:
        clean_architecture_config(config['pretraining']['architecture'])
        
    if 'applied_inference' in config and 'architecture' in config['applied_inference']:
        clean_architecture_config(config['applied_inference']['architecture'])
        
    if 'plasmode_experiments' in config:
        if 'generator_architecture' in config['plasmode_experiments']:
            clean_architecture_config(config['plasmode_experiments']['generator_architecture'])
        if 'evaluator_architecture' in config['plasmode_experiments']:
            clean_architecture_config(config['plasmode_experiments']['evaluator_architecture'])

    # Save new config
    target_config_path = target_exp_dir / "config.json"
    with open(target_config_path, 'w') as f:
        json.dump(config, f, indent=2)

print("Configuration generation complete.")
