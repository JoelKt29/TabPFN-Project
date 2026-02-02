"""
Benchmark Script: Compare Baseline TabPFN vs Phase 2 Models
Tests different configurations and reports comprehensive metrics
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN not installed")

from modified_architectures import CustomTabularTransformer, DeepFeedForward
from train_sabr_model import SABRModelTrainer


class ModelBenchmark:
    """Compare different models on SABR data."""
    
    def __init__(self, data_path: str, scaling_params_path: str = None):
        """
        Args:
            data_path: Path to SABR CSV data
            scaling_params_path: Path to scaling parameters JSON
        """
        self.data_path = data_path
        self.results = []
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Load scaling params
        self.scaling_params = None
        if scaling_params_path and Path(scaling_params_path).exists():
            with open(scaling_params_path, 'r') as f:
                self.scaling_params = json.load(f)
        
        # Prepare features and target
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare train/test split."""
        
        # Target
        if 'y_scaled' in self.df.columns:
            y = self.df['y_scaled'].values
        elif 'volatility_output' in self.df.columns:
            y = self.df['volatility_output'].values
        else:
            y = self.df['volatility'].values
        
        # Features
        exclude_cols = ['y_scaled', 'volatility_output', 'volatility']
        feature_cols = [c for c in self.df.columns 
                       if c not in exclude_cols and self.df[c].nunique() > 1]
        
        X = self.df[feature_cols].values
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        print(f"Features: {feature_cols}")
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        self.input_dim = X.shape[1]
    
    def _descale(self, y_scaled):
        """Descale predictions if scaling params available."""
        if self.scaling_params is not None and 'y_min' in self.scaling_params:
            y_min = self.scaling_params['y_min']
            y_max = self.scaling_params['y_max']
            return y_scaled * (y_max - y_min) + y_min
        return y_scaled
    
    def _compute_metrics(self, y_true, y_pred, model_name: str, train_time: float):
        """Compute and store metrics."""
        
        # Descale
        y_true_real = self._descale(y_true)
        y_pred_real = self._descale(y_pred)
        
        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        r2 = r2_score(y_true_real, y_pred_real)
        mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100
        
        result = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'train_time_sec': train_time
        }
        
        self.results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"MAE:        {mae:.8f}")
        print(f"RMSE:       {rmse:.8f}")
        print(f"R²:         {r2:.6f}")
        print(f"MAPE:       {mape:.4f}%")
        print(f"Train time: {train_time:.2f}s")
        print(f"{'='*60}")
    
    def benchmark_tabpfn(self):
        """Benchmark baseline TabPFN."""
        
        if not TABPFN_AVAILABLE:
            print("TabPFN not available, skipping...")
            return
        
        print("\n🔥 Benchmarking TabPFN (Baseline)...")
        
        start_time = time.time()
        
        try:
            regressor = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
            regressor.fit(self.X_train, self.y_train)
            predictions = regressor.predict(self.X_test)
            
            train_time = time.time() - start_time
            
            self._compute_metrics(
                self.y_test, predictions, 
                "TabPFN (Baseline)", train_time
            )
            
        except Exception as e:
            print(f"Error with TabPFN: {e}")
    
    def benchmark_custom_model(
        self,
        model_name: str,
        model: torch.nn.Module,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64
    ):
        """Benchmark a custom PyTorch model."""
        
        print(f"\n🔥 Benchmarking {model_name}...")
        
        start_time = time.time()
        
        # Prepare data
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train).reshape(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Prediction
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(device)
            predictions = model(X_test_tensor).cpu().numpy().flatten()
        
        train_time = time.time() - start_time
        
        self._compute_metrics(
            self.y_test, predictions,
            model_name, train_time
        )
    
    def run_all_benchmarks(self):
        """Run comprehensive benchmark suite."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL BENCHMARK")
        print("="*80)
        
        # 1. Baseline TabPFN
        self.benchmark_tabpfn()
        
        # 2. Transformer with different activations
        for activation in ['mish', 'gelu', 'swish', 'selu']:
            model = CustomTabularTransformer(
                input_dim=self.input_dim,
                d_model=128,
                nhead=4,
                num_encoder_layers=3,
                activation=activation,
                use_mlp_head=True
            )
            self.benchmark_custom_model(
                model_name=f"Transformer ({activation})",
                model=model,
                num_epochs=30,
                learning_rate=1e-3
            )
        
        # 3. Deep Feedforward with different activations
        for activation in ['mish', 'gelu']:
            model = DeepFeedForward(
                input_dim=self.input_dim,
                hidden_dims=[256, 128, 64],
                activation=activation
            )
            self.benchmark_custom_model(
                model_name=f"FeedForward ({activation})",
                model=model,
                num_epochs=30,
                learning_rate=1e-3
            )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print results summary table."""
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values('mae')
        
        print("\nRanked by MAE (lower is better):")
        print("-" * 80)
        print(f"{'Rank':<6} {'Model':<30} {'MAE':<12} {'RMSE':<12} {'R²':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for idx, row in df_results.iterrows():
            rank = df_results.index.get_loc(idx) + 1
            print(f"{rank:<6} {row['model']:<30} {row['mae']:<12.8f} "
                  f"{row['rmse']:<12.8f} {row['r2']:<10.6f} {row['train_time_sec']:<10.2f}")
        
        print("-" * 80)
        
        # Best model
        best = df_results.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['model']}")
        print(f"   MAE: {best['mae']:.8f}")
        print(f"   Improvement over baseline: ", end="")
        
        if 'TabPFN (Baseline)' in df_results['model'].values:
            baseline_mae = df_results[df_results['model'] == 'TabPFN (Baseline)']['mae'].values[0]
            improvement = ((baseline_mae - best['mae']) / baseline_mae) * 100
            print(f"{improvement:.2f}%")
        else:
            print("N/A (no baseline)")
        
        print("="*80)
        
        # Save results
        output_file = 'benchmark_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
        
        return df_results


def main():
    """Run the benchmark."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark SABR models')
    parser.add_argument('--data', type=str, default='sabr_data_recovery.csv',
                        help='Path to SABR data')
    parser.add_argument('--scaling', type=str, default='scaling_params_recovery.json',
                        help='Path to scaling parameters')
    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmark (fewer models, fewer epochs)')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ModelBenchmark(
        data_path=args.data,
        scaling_params_path=args.scaling
    )
    
    if args.quick:
        print("Running QUICK benchmark (TabPFN + 2 models)...")
        benchmark.benchmark_tabpfn()
        
        model = CustomTabularTransformer(
            input_dim=benchmark.input_dim,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            activation='mish'
        )
        benchmark.benchmark_custom_model("Transformer (Mish)", model, num_epochs=20)
        
        model = DeepFeedForward(
            input_dim=benchmark.input_dim,
            hidden_dims=[256, 128],
            activation='gelu'
        )
        benchmark.benchmark_custom_model("FeedForward (GELU)", model, num_epochs=20)
        
        benchmark.print_summary()
    else:
        benchmark.run_all_benchmarks()


if __name__ == "__main__":
    main()
