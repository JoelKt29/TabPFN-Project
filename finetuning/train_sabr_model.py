"""
Complete TabPFN Fine-Tuning Pipeline for SABR Data
Includes training with derivatives, custom losses, and evaluation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Optional, Tuple

from modified_architectures import CustomTabularTransformer, DeepFeedForward
from custom_losses import SABRDerivativeLoss, WeightedMAELoss, HuberLossWithDerivatives


class SABRModelTrainer:
    """
    Complete training pipeline for SABR volatility prediction.
    Handles both value-only and value+derivative training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        checkpoint_dir: str = './checkpoints'
    ):
        """
        Args:
            model: PyTorch model to train
            device: 'auto', 'cuda', or 'cpu'
            checkpoint_dir: Where to save model checkpoints
        """
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rates': []
        }
    
    def prepare_data(
        self,
        data_path: str,
        train_split: float = 0.8,
        batch_size: int = 64,
        use_greeks: bool = False,
        scaling_params_path: Optional[str] = None
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        """
        Load and prepare data for training.
        
        Returns:
            train_loader, val_loader, data_info
        """
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
        
        # Load scaling parameters if provided
        scaling_params = None
        if scaling_params_path and Path(scaling_params_path).exists():
            with open(scaling_params_path, 'r') as f:
                scaling_params = json.load(f)
        
        # Prepare features and targets
        if use_greeks:
            # Multi-output: volatility + all Greeks
            target_cols = ['volatility'] + [c for c in df.columns if 'dV_' in c or 'd2V_' in c]
            if not all(col in df.columns for col in target_cols):
                raise ValueError("Greeks not found in data. Run sabr_derivatives.py first.")
            y = df[target_cols].values
            print(f"Using {len(target_cols)} outputs (vol + {len(target_cols)-1} Greeks)")
        else:
            # Single output: volatility only
            if 'y_scaled' in df.columns:
                y = df['y_scaled'].values.reshape(-1, 1)
            elif 'volatility_output' in df.columns:
                y = df['volatility_output'].values.reshape(-1, 1)
            elif 'volatility' in df.columns:
                y = df['volatility'].values.reshape(-1, 1)
            else:
                raise ValueError("No volatility column found")
            print("Using single output (volatility only)")
        
        # Features (exclude targets and constants)
        exclude_cols = set(['y_scaled', 'volatility_output', 'volatility'])
        if use_greeks:
            exclude_cols.update([c for c in df.columns if 'dV_' in c or 'd2V_' in c])
        
        feature_cols = [c for c in df.columns 
                       if c not in exclude_cols and df[c].nunique() > 1]
        
        X = df[feature_cols].values
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Create datasets
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )
        
        # Split
        n_train = int(len(dataset) * train_split)
        n_val = len(dataset) - n_train
        
        train_dataset, val_dataset = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        data_info = {
            'input_dim': X.shape[1],
            'output_dim': y.shape[1],
            'feature_cols': feature_cols,
            'n_train': n_train,
            'n_val': n_val,
            'scaling_params': scaling_params
        }
        
        return train_loader, val_loader, data_info
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ):
        """
        Train the model.
        """
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss, val_mae = self.validate(val_loader, criterion)
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
            
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Val MAE: {val_mae:.6f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validation step."""
        self.model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                mae = torch.mean(torch.abs(outputs - batch_y))
                
                val_loss += loss.item()
                val_mae += mae.item()
        
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        return val_loss, val_mae
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        scaling_params: Optional[Dict] = None,
        save_plots: bool = True
    ) -> Dict:
        """
        Comprehensive evaluation with metrics and plots.
        """
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        
        # Descale if needed
        if scaling_params is not None and 'y_min' in scaling_params:
            y_min = scaling_params['y_min']
            y_max = scaling_params['y_max']
            predictions = predictions * (y_max - y_min) + y_min
            targets = targets * (y_max - y_min) + y_min
        
        # Compute metrics (for first output - volatility)
        pred_vol = predictions[:, 0]
        true_vol = targets[:, 0]
        
        metrics = {
            'mae': mean_absolute_error(true_vol, pred_vol),
            'rmse': np.sqrt(mean_squared_error(true_vol, pred_vol)),
            'r2': r2_score(true_vol, pred_vol),
            'mape': np.mean(np.abs((true_vol - pred_vol) / true_vol)) * 100
        }
        
        print("\n" + "="*60)
        print("EVALUATION METRICS (Volatility)")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric.upper():8s}: {value:.6f}")
        print("="*60)
        
        # Plots
        if save_plots:
            self.plot_results(predictions, targets, metrics)
        
        return metrics
    
    def plot_results(self, predictions: np.ndarray, targets: np.ndarray, metrics: Dict):
        """Generate evaluation plots."""
        
        pred_vol = predictions[:, 0]
        true_vol = targets[:, 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Predicted vs Actual
        axes[0, 0].scatter(true_vol, pred_vol, alpha=0.5, s=10)
        axes[0, 0].plot([true_vol.min(), true_vol.max()],
                        [true_vol.min(), true_vol.max()],
                        'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('True Volatility')
        axes[0, 0].set_ylabel('Predicted Volatility')
        axes[0, 0].set_title(f'Predicted vs True (R²={metrics["r2"]:.4f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = pred_vol - true_vol
        axes[0, 1].scatter(true_vol, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('True Volatility')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title(f'Residual Plot (MAE={metrics["mae"]:.6f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual histogram
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training history
        if len(self.history['train_loss']) > 0:
            epochs = range(1, len(self.history['train_loss']) + 1)
            axes[1, 1].plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
            axes[1, 1].plot(epochs, self.history['val_loss'], label='Val Loss', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training History')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = self.checkpoint_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
        plt.close()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']


def main():
    """Example usage."""
    
    print("="*80)
    print("SABR MODEL FINE-TUNING PIPELINE")
    print("="*80)
    
    # Configuration
    config = {
        'data_path': 'sabr_data_recovery.csv',
        'scaling_params_path': 'scaling_params_recovery.json',
        'use_greeks': False,  # Set to True if you have derivative data
        'model_type': 'transformer',  # or 'feedforward'
        'activation': 'mish',
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'device': 'auto'
    }
    
    # Create model
    print("\nCreating model...")
    if config['model_type'] == 'transformer':
        model = CustomTabularTransformer(
            input_dim=8,  # Will be updated after loading data
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            activation=config['activation'],
            use_mlp_head=True,
        )
    else:
        model = DeepFeedForward(
            input_dim=8,
            hidden_dims=[512, 256, 128, 64],
            activation=config['activation']
        )
    
    # Create trainer
    trainer = SABRModelTrainer(model, device=config['device'])
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, data_info = trainer.prepare_data(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        use_greeks=config['use_greeks'],
        scaling_params_path=config['scaling_params_path']
    )
    
    # Recreate model with correct dimensions
    if config['model_type'] == 'transformer':
        model = CustomTabularTransformer(
            input_dim=data_info['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            activation=config['activation'],
            output_dim=data_info['output_dim'],
            use_mlp_head=True,
        )
    else:
        model = DeepFeedForward(
            input_dim=data_info['input_dim'],
            hidden_dims=[512, 256, 128, 64],
            output_dim=data_info['output_dim'],
            activation=config['activation']
        )
    
    trainer = SABRModelTrainer(model, device=config['device'])
    
    # Setup training
    criterion = nn.L1Loss()  # MAE loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Train
    print("\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        scheduler=scheduler,
        early_stopping_patience=20
    )
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(
        test_loader=val_loader,
        scaling_params=data_info['scaling_params'],
        save_plots=True
    )
    
    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
