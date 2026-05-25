
from pytris import API
import pandas as pd
import plotly.express as px
import os
import numpy as np
import torch.optim as optim
import torch 
from Traffic import TrafficDataset, ModelTrainer, TrafficDataProcessor, TrafficNet
from collections import Counter


pd.set_option('future.no_silent_downcasting', False)
# Set the option to ignore chained assignment warnings
pd.options.mode.chained_assignment = None
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../scripts") 




# Add these diagnostic functions to identify and fix remaining issues

def diagnose_data_quality(processor):
    """Diagnose potential data leakage and quality issues"""
    df = processor.df
    
    print("=== DATA QUALITY DIAGNOSIS ===")
    
    # 1. Check for data leakage
    print("\n1. POTENTIAL DATA LEAKAGE:")
    corr_with_target = df.corr()['congestion'].sort_values(key=abs, ascending=False)
    print("Correlations with target:")
    for feature, corr in corr_with_target.items():
        if feature != 'congestion' and abs(corr) > 0.8:
            print(f"  ⚠️  HIGH CORRELATION: {feature}: {corr:.3f}")
        elif feature != 'congestion':
            print(f"  ✓  {feature}: {corr:.3f}")
    
    # 2. Check class distribution over time
    print(f"\n2. CLASS DISTRIBUTION:")
    total_samples = len(df)
    congestion_samples = df['congestion'].sum()
    print(f"  Total samples: {total_samples}")
    print(f"  Congestion samples: {congestion_samples} ({100*congestion_samples/total_samples:.2f}%)")
    
    # 3. Check temporal distribution
    print(f"\n3. TEMPORAL DISTRIBUTION:")
    df_monthly = df.groupby(df.index.to_period('M'))['congestion'].agg(['count', 'sum', 'mean'])
    print("Monthly congestion rates:")
    for month, stats in df_monthly.iterrows():
        print(f"  {month}: {stats['sum']}/{stats['count']} samples ({100*stats['mean']:.1f}%)")
    
    # 4. Check for obvious patterns
    print(f"\n4. PATTERN ANALYSIS:")
    hourly_congestion = df.groupby('hour')['congestion'].mean()
    peak_hours = hourly_congestion[hourly_congestion > hourly_congestion.mean() + hourly_congestion.std()]
    print(f"  Peak congestion hours: {list(peak_hours.index)}")
    
    daily_congestion = df.groupby('dayofweek')['congestion'].mean()
    peak_days = daily_congestion[daily_congestion > daily_congestion.mean() + daily_congestion.std()]
    print(f"  Peak congestion days: {list(peak_days.index)} (0=Monday)")


def create_better_splits(processor, seq_len=48, splits=(0.7, 0.15, 0.15)):
    """Create more robust train/val/test splits"""
    df = processor.df
    target_col = "congestion"
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Ensure we have enough samples for each split to be representative
    n_rows = len(df)
    n_samples = n_rows - seq_len
    
    print(f"=== SPLIT ANALYSIS ===")
    print(f"Total rows: {n_rows}, Available samples: {n_samples}")
    
    # Calculate split sizes
    n_train = int(n_samples * splits[0])
    n_val = int(n_samples * splits[1])
    n_test = n_samples - n_train - n_val
    
    print(f"Train samples: {n_train}, Val samples: {n_val}, Test samples: {n_test}")
    
    # Check class distribution in each split
    train_end_row = n_train + seq_len
    val_start_row = n_train
    val_end_row = n_train + n_val + seq_len
    test_start_row = n_train + n_val
    
    # Analyze splits
    splits_info = {}
    for split_name, start_idx, end_idx in [
        ("train", 0, train_end_row),
        ("val", val_start_row, val_end_row), 
        ("test", test_start_row, n_rows)
    ]:
        split_df = df.iloc[start_idx:end_idx]
        congestion_count = split_df['congestion'].sum()
        total_count = len(split_df)
        congestion_rate = congestion_count / total_count if total_count > 0 else 0
        
        splits_info[split_name] = {
            'samples': total_count - seq_len if split_name != 'train' else n_train,
            'congestion_count': congestion_count,
            'congestion_rate': congestion_rate,
            'date_range': (split_df.index.min(), split_df.index.max())
        }
        
        print(f"\n{split_name.upper()} SET:")
        print(f"  Samples: {splits_info[split_name]['samples']}")
        print(f"  Congestion: {congestion_count}/{total_count} ({100*congestion_rate:.2f}%)")
        print(f"  Date range: {splits_info[split_name]['date_range'][0]} to {splits_info[split_name]['date_range'][1]}")
    
    return splits_info


def create_conservative_model(input_size):
    """Create a more conservative model to prevent overfitting"""
    return TrafficNet(
        input_size=input_size,
        hidden_size=16,      # Very small hidden size
        num_layers=1,        # Single layer
        dropout=0.5          # High dropout
    )


def create_baseline_comparison(processor, splits_info):
    """Create simple baselines to compare against LSTM"""
    df = processor.df
    
    print("=== BASELINE COMPARISONS ===")
    
    # 1. Always predict majority class
    total_samples = sum(info['samples'] for info in splits_info.values())
    majority_accuracy = (total_samples - sum(info['congestion_count'] for info in splits_info.values())) / total_samples
    print(f"1. Majority class baseline accuracy: {majority_accuracy:.4f}")
    
    # 2. Time-based rules
    df_with_time = df.copy()
    
    # Simple rule: congestion during rush hours on weekdays
    rush_hours = [7, 8, 9, 17, 18, 19]  # 7-9 AM, 5-7 PM
    weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
    
    rule_based_pred = (
        (df_with_time['hour'].isin(rush_hours)) & 
        (df_with_time['dayofweek'].isin(weekdays))
    ).astype(int)
    
    rule_accuracy = (rule_based_pred == df_with_time['congestion']).mean()
    print(f"2. Rush hour rule accuracy: {rule_accuracy:.4f}")
    
    # 3. Speed threshold baseline
    speed_threshold_pred = (df_with_time['Avg mph'] < 25).astype(int)
    threshold_accuracy = (speed_threshold_pred == df_with_time['congestion']).mean()
    print(f"3. Speed threshold baseline accuracy: {threshold_accuracy:.4f}")
    
    return {
        'majority_baseline': majority_accuracy,
        'rule_based': rule_accuracy, 
        'speed_threshold': threshold_accuracy
    }


def improved_training_setup(model, class_weights, patience=15):
    """Set up more robust training configuration"""
    
    # More conservative optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.0005,           # Lower learning rate
        weight_decay=1e-4,   # L2 regularization
        eps=1e-8
    )
    
    # More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,          # Reduce by half
        patience=5,          # Wait 5 epochs
        min_lr=1e-6,
        verbose=True
    )
    
    # Weighted loss with smoother weights
    if class_weights is not None:
        # Cap the weight ratio to prevent extreme weighting
        max_weight_ratio = 10
        min_weight = min(class_weights.values())
        capped_weights = {k: min(v, min_weight * max_weight_ratio) for k, v in class_weights.items()}
        weight_tensor = torch.tensor([capped_weights[0], capped_weights[1]], dtype=torch.float32)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    return optimizer, scheduler, criterion


# Usage example with diagnostics:
def run_diagnostics_and_retrain(processor, seq_len=48):
    """Run full diagnostic pipeline and retrain with fixes"""
    
    # 1. Diagnose data quality
    diagnose_data_quality(processor)
    
    # 2. Analyze splits
    splits_info = create_better_splits(processor, seq_len)
    
    # 3. Create baseline comparisons
    baselines = create_baseline_comparison(processor, splits_info)
    
    # 4. Create conservative model
    feature_cols = [c for c in processor.df.columns if c != 'congestion']
    model = create_conservative_model(len(feature_cols))
    
    print(f"\n=== MODEL ARCHITECTURE ===")
    print(f"Input size: {len(feature_cols)}")
    print(f"Hidden size: 16 (very conservative)")
    print(f"Layers: 1 (minimal complexity)")
    print(f"Dropout: 0.5 (high regularization)")
    
    # 5. Create data loaders with less aggressive sampling
    train_loader, val_loader, test_loader, info = processor.create_loaders_from_df(
        seq_len=seq_len, 
        handle_imbalance=True, 
        batch_size=32,  # Smaller batch size
        splits=(0.7, 0.2, 0.1)  # Larger validation set
    )
    
    # 6. Setup improved training
    train_labels = []
    for features, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    trainer = ModelTrainer(model, class_weights)
    trainer.optimizer, trainer.scheduler, trainer.criterion = improved_training_setup(
        model, class_weights, patience=15
    )
    
    print(f"\n=== TRAINING SETUP ===")
    print(f"Class weights: {class_weights}")
    print(f"Learning rate: 0.0005")
    print(f"Early stopping patience: 15")
    
    return trainer, train_loader, val_loader, test_loader, baselines

# After running diagnostics, compare your LSTM results with baselines:
# If LSTM accuracy < baseline + 0.05, consider using simpler approaches