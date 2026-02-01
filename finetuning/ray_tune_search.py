"""
Ray Tune Configuration for SABR Model Hyperparameter Search

This script sets up automated hyperparameter optimization using Ray Tune.
It will search over:
- Activation functions (Swish, Mish, GELU, SELU)
- Network architectures (depth, width)
- Learning rates and optimizers
- Loss function configurations
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from typing import Dict, Any

try:
    from ray import tune, train
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray Tune not installed. Install with: pip install ray[tune] optuna")

from modified_architectures import CustomTabularTransformer, DeepFeedForward
from custom_losses import SABRDerivativeLoss, WeightedMAELoss


def load_data(data_path: str, use_greeks: bool = False):
    """Load and prepare SABR data."""
    df = pd.read_csv(data_path)
    
    # Separate features and targets
    if use_greeks:
        # Use volatility + Greeks as targets
        target_cols = ['volatility'] + [c for c in df.columns if 'dV_' in c or 'd2V_' in c]
        y = df[target_cols].values
    else:
        # Just volatility
        if 'y_scaled' in df.columns:
            y = df['y_scaled'].values
        elif 'volatility_output' in df.columns:
            y = df['volatility_output'].values
        else:
            y = df['volatility'].values
    
    # Features (exclude targets and constants)
    exclude_cols = target_cols if use_greeks else ['y_scaled', 'volatility_output', 'volatility']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].nunique() > 1]
    
    X = df[feature_cols].values
    
    return X, y, feature_cols


def create_dataloaders(X, y, batch_size=64, train_split=0.8):
    """Create PyTorch dataloaders."""
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def train_model(config: Dict[str, Any], data_path: str, checkpoint_dir: str = None):
    """
    Training function for Ray Tune.
    
    Args:
        config: Hyperparameter configuration from Ray Tune
        data_path: Path to data CSV
        checkpoint_dir: Directory for checkpoints
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    X, y, feature_cols = load_data(data_path, use_greeks=config.get('use_greeks', False))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X, y,
        batch_size=config['batch_size'],
        train_split=0.8
    )
    
    # Create model
    input_dim = X.shape[1]
    output_dim = 1 if len(y.shape) == 1 else y.shape[1]
    
    if config['model_type'] == 'transformer':
        model = CustomTabularTransformer(
            input_dim=input_dim,
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation=config['activation'],
            output_dim=output_dim,
            use_mlp_head=config['use_mlp_head'],
            mlp_hidden_dims=config.get('mlp_hidden_dims', None),
        )
    else:  # feedforward
        model = DeepFeedForward(
            input_dim=input_dim,
            hidden_dims=config['hidden_dims'],
            output_dim=output_dim,
            activation=config['activation'],
            dropout=config['dropout'],
        )
    
    model = model.to(device)
    
    # Loss function
    if config['loss_type'] == 'mae':
        criterion = nn.L1Loss()
    elif config['loss_type'] == 'mse':
        criterion = nn.MSELoss()
    elif config['loss_type'] == 'huber':
        criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    elif config['loss_type'] == 'derivative':
        criterion = SABRDerivativeLoss(
            value_weight=config.get('value_weight', 1.0),
            derivative_weight=config.get('derivative_weight', 0.5)
        )
    else:
        criterion = nn.L1Loss()
    
    # Optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['lr'],
            momentum=config.get('momentum', 0.9)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    num_epochs = config.get('num_epochs', 50)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                loss = criterion(outputs, batch_y)
                mae = torch.mean(torch.abs(outputs - batch_y))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Report to Ray Tune
        train.report({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'epoch': epoch
        })


def run_ray_tune_search(
    data_path: str,
    num_samples: int = 50,
    max_epochs: int = 50,
    gpus_per_trial: float = 0.0,
    cpus_per_trial: float = 1.0,
    output_dir: str = './ray_results'
):
    """
    Run hyperparameter search with Ray Tune.
    
    Args:
        data_path: Path to SABR data CSV
        num_samples: Number of configurations to try
        max_epochs: Maximum epochs per trial
        gpus_per_trial: GPUs per trial (0 for CPU)
        cpus_per_trial: CPUs per trial
        output_dir: Where to save results
    """
    
    if not RAY_AVAILABLE:
        raise ImportError("Ray Tune not available. Install with: pip install ray[tune] optuna")
    
    # Define search space
    search_space = {
        # Model architecture
        'model_type': tune.choice(['transformer', 'feedforward']),
        'activation': tune.choice(['swish', 'mish', 'gelu', 'selu']),
        
        # Transformer-specific
        'd_model': tune.choice([128, 256, 512]),
        'nhead': tune.choice([4, 8]),
        'num_layers': tune.choice([2, 3, 4, 6]),
        'dim_feedforward': tune.choice([512, 1024, 2048]),
        'use_mlp_head': tune.choice([True, False]),
        
        # Feedforward-specific
        'hidden_dims': tune.choice([
            [256, 128, 64],
            [512, 256, 128],
            [512, 256, 128, 64],
            [1024, 512, 256],
        ]),
        
        # Training hyperparameters
        'batch_size': tune.choice([32, 64, 128]),
        'lr': tune.loguniform(1e-5, 1e-2),
        'dropout': tune.uniform(0.0, 0.3),
        'optimizer': tune.choice(['adam', 'adamw']),
        'weight_decay': tune.loguniform(1e-6, 1e-3),
        
        # Loss configuration
        'loss_type': tune.choice(['mae', 'huber', 'derivative']),
        'value_weight': tune.uniform(0.5, 1.5),
        'derivative_weight': tune.uniform(0.1, 1.0),
        'huber_delta': tune.loguniform(1e-3, 1.0),
        
        # Data
        'use_greeks': tune.choice([False, True]),
        'num_epochs': max_epochs,
    }
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr='epoch',
        metric='val_mae',
        mode='min',
        max_t=max_epochs,
        grace_period=10,
        reduction_factor=2,
    )
    
    # Optuna search algorithm
    search_alg = OptunaSearch(
        metric='val_mae',
        mode='min',
    )
    
    # Run tuning
    print(f"Starting Ray Tune search with {num_samples} trials...")
    print(f"Results will be saved to: {output_dir}")
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model, data_path=data_path),
            resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
        ),
        param_space=search_space,
        run_config=train.RunConfig(
            name="sabr_hyperparam_search",
            storage_path=output_dir,
        )
    )
    
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(metric="val_mae", mode="min")
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND:")
    print("="*80)
    print(f"Best MAE: {best_result.metrics['val_mae']:.6f}")
    print(f"\nBest config:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config_path = os.path.join(output_dir, 'best_config.json')
    with open(best_config_path, 'w') as f:
        json.dump(best_result.config, f, indent=4)
    
    print(f"\nBest config saved to: {best_config_path}")
    
    return results, best_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ray Tune hyperparameter search for SABR')
    parser.add_argument('--data', type=str, default='sabr_data_recovery.csv',
                        help='Path to SABR data CSV')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of trials to run')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per trial')
    parser.add_argument('--gpus', type=float, default=0.0,
                        help='GPUs per trial (0 for CPU)')
    parser.add_argument('--cpus', type=float, default=1.0,
                        help='CPUs per trial')
    parser.add_argument('--output', type=str, default='./ray_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Run search
    results, best_result = run_ray_tune_search(
        data_path=args.data,
        num_samples=args.samples,
        max_epochs=args.epochs,
        gpus_per_trial=args.gpus,
        cpus_per_trial=args.cpus,
        output_dir=args.output,
    )
    
    print("\n✅ Ray Tune search completed!")
