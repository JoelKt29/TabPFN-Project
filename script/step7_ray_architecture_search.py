import os
os.environ['RAY_DEDUP_LOGS'] = '0'  # Reduce Ray logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from pathlib import Path
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from step6_loss_with_derivatives import DerivativeLoss
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split




current_dir = Path(__file__).resolve().parent
data_dir = current_dir.parent / "data"
script_dir = str(current_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


def get_activation(name: str):
    activations = {
        'swish': Swish(),
        'mish': Mish(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),}
    return activations[name.lower()]



class TabularTransformer(nn.Module):
    """
    Inspired by TabPFN architecture
    """
    
    def __init__(
        self,
        input_dim: int=8,
        output_dim: int = 7,           
        d_model: int = 192,            
        nhead: int = 4,                
        num_layers: int = 12,           
        dim_feedforward: int = 768,   
        dropout: float = 0.0,          
        activation: str = 'gelu'):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(nn.Linear(d_model, d_model // 2),
            get_activation(activation), nn.Dropout(dropout),nn.Linear(d_model // 2, output_dim))
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)  
        x = x.unsqueeze(1)  
        x = x + self.pos_encoding
        x = self.transformer(x)  
        x = x.squeeze(1)  
        x = self.output_proj(x) 
        return x



class DeepMLP(nn.Module):    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = 'gelu',
        d_model : int = 0 ):

        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim),nn.BatchNorm1d(hidden_dim),
                get_activation(activation),nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)



def train_model_ray(config: dict):
    """
    Training a model with an input configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    raw_path = config.get('data_path',data_dir / 'sabr_with_derivatives_scaled.csv')
    data_path = os.path.abspath(raw_path)
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        df = pd.read_csv(raw_path)


    
    feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
    has_derivatives = 'dV_dbeta_scaled' in df.columns
    if has_derivatives:
        deriv_cols = [c for c in df.columns if c.endswith('_scaled') and c.startswith('dV_')]
        output_dim = 1 + len(deriv_cols)  # volatility + derivatives
    else:
        output_dim = 1
    X = df[feature_cols].values
    if has_derivatives:
        y_cols = ['volatility_scaled'] + deriv_cols
        y = df[y_cols].values
    else:
        y = df['y_scaled'].values.reshape(-1, 1) if 'y_scaled' in df else df['volatility_output'].values.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])


    model = None
    if config["model_type"] == "transformer":
        # PyTorch Transformer only accepts relu and gelu
        if config["activation"] not in ["gelu", "relu"]:
            model = DeepMLP(input_dim=X.shape[1], output_dim=output_dim,hidden_dims=config["hidden_dims"], 
                            dropout=config.get("dropout", 0.1),activation=config["activation"] )
        else:
            model = TabularTransformer(input_dim=X.shape[1],output_dim=output_dim,d_model=config["d_model"],
                nhead=config["nhead"],num_layers=config["num_layers"],im_feedforward=config["dim_feedforward"],
                dropout=config.get("dropout", 0.1),activation=config["activation"])
    elif config["model_type"] == "mlp":
        model = DeepMLP(input_dim=X.shape[1],output_dim=output_dim,hidden_dims=config["hidden_dims"],
            dropout=config.get("dropout", 0.1),activation=config["activation"])
    model = model.to(device)
    

    if has_derivatives and config.get('use_derivative_loss', True):
        criterion = DerivativeLoss(value_weight=1.0, derivative_weight=0.05)
    else:
        criterion = nn.L1Loss()
    

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    if config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-5))
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    num_epochs = config.get('num_epochs', 50)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if has_derivatives and config.get('use_derivative_loss', True):
                pred_vol = outputs[:, 0:1]
                true_vol = batch_y[:, 0:1]
                pred_derivs = {f'deriv_{i}': outputs[:, i:i+1] for i in range(1, outputs.size(1))}
                true_derivs = {f'deriv_{i}': batch_y[:, i:i+1] for i in range(1, batch_y.size(1))}
                loss, _ = criterion(pred_vol, true_vol, pred_derivs, true_derivs)
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                if has_derivatives and config.get('use_derivative_loss', True):
                    pred_vol = outputs[:, 0:1]
                    true_vol = batch_y[:, 0:1]
                    pred_derivs = {f'deriv_{i}': outputs[:, i:i+1] for i in range(1, outputs.size(1))}
                    true_derivs = {f'deriv_{i}': batch_y[:, i:i+1] for i in range(1, batch_y.size(1))}
                    loss, _ = criterion(pred_vol, true_vol, pred_derivs, true_derivs)
                    mae = torch.mean(torch.abs(pred_vol - true_vol))
                else:
                    loss = criterion(outputs, batch_y)
                    mae = torch.mean(torch.abs(outputs - batch_y))  
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        scheduler.step(val_loss)
        
        tune.report({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'epoch': epoch
        })


def run_ray_tune_search(
    data_path: str = data_dir / 'sabr_with_derivatives_scaled.csv',
    num_samples: int = 100,
    max_epochs: int = 50,
    gpus_per_trial: float = 1.0,
    output_dir: str = './ray_results'
):
    
    
    search_space = {
        'data_path': os.path.abspath(data_path),
        'model_type': tune.choice(['transformer', 'mlp']),
        'activation': tune.choice(['swish', 'mish', 'gelu', 'selu']),  # All differentiable
        
        # Transformer-specific
        'd_model': tune.choice([128, 256, 512]),
        'nhead': tune.choice([4, 8]),
        'num_layers': tune.choice([2, 3, 4, 6]),
        'dim_feedforward': tune.choice([512, 1024, 2048]),
        
        # MLP-specific
        'hidden_dims': tune.choice([(512, 256, 128), (256, 128, 64)]),
        
        # Training
        'batch_size': tune.choice([32, 64, 128]),
        'lr': tune.loguniform(1e-5, 1e-2),
        'dropout': tune.uniform(0.0, 0.3),
        'optimizer': tune.choice(['adam', 'adamw']),
        'weight_decay': tune.loguniform(1e-6, 1e-3),
        'num_epochs': max_epochs,

        # Loss
        'use_derivative_loss':True,
        'value_weight': tune.uniform(0.5, 1.5),
        'derivative_weight': tune.uniform(0.1, 1.0),
        
    }
    
    scheduler = ASHAScheduler(time_attr='epoch',metric='val_loss',mode='min',
        max_t=max_epochs,grace_period=20,reduction_factor=2,)
    search_alg = OptunaSearch(metric='val_loss',mode='min',)
    
    print("="*80)
    print("STARTING RAY TUNE ARCHITECTURE SEARCH")
    print("="*80)
    

    abs_output_dir = os.path.abspath(output_dir)
    ray.init(ignore_reinit_error=True)
    tuner = tune.Tuner(
        tune.with_resources(
            train_model_ray,
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples,
        ),
        param_space=search_space,
        )
    
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND")
    print("="*80)
    print(f"Best MAE: {best_result.metrics['val_mae']:.8f}")
    print(f"\nBest configuration:")
    for key, value in sorted(best_result.config.items()):
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    
    # Save best config
    best_config_path = Path(output_dir) / 'best_config.json'
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_to_save = best_result.config.copy()
    best_config_to_save['best_mae'] = float(best_result.metrics['val_mae'])
    with open(best_config_path, 'w') as f:
        json.dump(best_result.config, f, indent=2)
    return results, best_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ray Tune architecture search for SABR TabPFN')
    parser.add_argument('--data', type=str, default=data_dir / 'sabr_with_derivatives_scaled.csv',
                        help='Path to data CSV')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of configurations to try')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per trial')
    parser.add_argument('--gpus', type=float, default=1.0,
                        help='GPU fraction per trial (0 for CPU only)')
    parser.add_argument('--output', type=str, default='./ray_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    
    # Run search
    results, best_result = run_ray_tune_search(
        data_path=args.data,
        num_samples=args.samples,
        max_epochs=args.epochs,
        gpus_per_trial=args.gpus,
        output_dir=args.output
    )
    
    print("\n Ray Tune search completed")