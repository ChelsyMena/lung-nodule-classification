import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pathlib import Path

def format_config(config_dict):
    """Format configuration dictionary in a readable way"""
    output = []
    for key, value in config_dict.items():
        # Handle Path objects
        if isinstance(value, Path):
            value = str(value)
        # Skip private attributes
        if not key.startswith('_'):
            output.append(f"{key}: {value}")
    return "\n\t".join(output)

def display_training_metrics(results_dir="results"):
    results_path = Path(results_dir)
    
    # Find all config.npy files
    config_files = list(results_path.glob("*/config.npy"))
    
    print(f"\nFound {len(config_files)} training runs")
    print("-" * 50)
    
    for config_file in config_files:
        experiment_name = config_file.parent.name
        try:
            metrics = np.load(config_file, allow_pickle=True).item()
            
            print(f"📊 Experiment: {experiment_name}")
            print(f"🎯 Best AUC: {metrics.get('best_auc', 'N/A'):.4f}")
            print(f"⏱️ Best Epoch: {metrics.get('epoch', 'N/A')}")
            print(f"📈 Memory Usage: {metrics.get('memory_usage_mb', 'N/A')}")
            print(f"📅 Training Time: {metrics.get('total_training_time', 'N/A')}")
            
            if 'config' in metrics:
                print("\n📋 Configuration:")
                print(f"\t{format_config(metrics['config'].__dict__)}")
            
            print("-" * 80 + "\n")
        except Exception as e:
            print(f"❌ Error reading {experiment_name}: {str(e)}\n")


if __name__ == "__main__":
    display_training_metrics()