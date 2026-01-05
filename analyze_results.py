import pandas as pd
from pathlib import Path
import os
import json
import glob

# Define base path for results
RESULTS_BASE = Path("../pcori_experiments/rethink_confounders_real_data/experiments_1-5-26")
OUTPUT_FILE = RESULTS_BASE / "all_experiments_summary.csv"

def analyze_all_experiments():
    all_summaries = []
    
    # Find all experiment directories
    exp_dirs = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir() and d.name.startswith("exp")])
    
    print(f"Found {len(exp_dirs)} experiment directories.")
    
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        print(f"Processing {exp_name}...")
        
        results_dir = exp_dir / "results"
        plasmode_results_path = results_dir / "plasmode_experiments" / "results.csv"
        
        if not plasmode_results_path.exists():
            print(f"  Warning: No plasmode results found for {exp_name}")
            continue
            
        try:
            # Read plasmode results
            df = pd.read_csv(plasmode_results_path)
            
            # Group by generation mode and calculate metrics
            summary = df.groupby('generation_mode').agg({
                'ate_bias': ['mean', 'std'],
                'ate_rmse': ['mean'],
                'ite_correlation': ['mean', 'std'],
                'ite_regression_slope': ['mean']
            }).reset_index()
            
            # Flatten columns
            summary.columns = ['_'.join(col).strip() if col[1] else col[0] for col in summary.columns.values]
            
            # Add experiment name
            summary.insert(0, 'experiment', exp_name)
            
            all_summaries.append(summary)
            
        except Exception as e:
            print(f"  Error processing {exp_name}: {e}")
    
    if all_summaries:
        final_df = pd.concat(all_summaries, ignore_index=True)
        print("\nAnalysis Complete.")
        print("-" * 80)
        print(final_df.to_string())
        print("-" * 80)
        
        # Save to CSV
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Summary saved to: {OUTPUT_FILE}")
    else:
        print("No results found to analyze.")

if __name__ == "__main__":
    analyze_all_experiments()
