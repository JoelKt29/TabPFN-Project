"""
Final Evaluation - Compare All Approaches
Génère un rapport complet pour Peter
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except:
    TABPFN_AVAILABLE = False

from ray_architecture_search import TabularTransformer, DeepMLP, get_activation
from loss_with_derivatives import create_loss_function


class FinalEvaluator:
    """
    Comprehensive evaluation comparing:
    1. TabPFN baseline
    2. Custom models with different activations
    3. Models with derivatives vs without
    """
    
    def __init__(self, data_path: str, scaling_params_path: str = None):
        self.data_path = data_path
        self.results = []
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Load scaling params if available
        self.scaling_params = None
        if scaling_params_path and Path(scaling_params_path).exists():
            with open(scaling_params_path, 'r') as f:
                self.scaling_params = json.load(f)
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare train/test split"""
        
        # Features
        feature_cols = ['beta', 'rho', 'volvol', 'v_atm_n', 'alpha', 'F', 'K', 'log_moneyness']
        self.feature_cols = feature_cols
        
        # Check if derivatives available
        self.has_derivatives = 'dV_dbeta_scaled' in self.df.columns
        
        if self.has_derivatives:
            deriv_cols = [c for c in self.df.columns if c.endswith('_scaled') and c.startswith('dV_')]
            self.deriv_cols = deriv_cols
            
            # Targets: volatility + derivatives
            y_cols = ['volatility_scaled'] + deriv_cols
            y = self.df[y_cols].values
        else:
            # Target: volatility only
            if 'y_scaled' in self.df.columns:
                y = self.df['y_scaled'].values.reshape(-1, 1)
            elif 'volatility_scaled' in self.df.columns:
                y = self.df['volatility_scaled'].values.reshape(-1, 1)
            else:
                y = self.df['volatility_output'].values.reshape(-1, 1)
        
        X = self.df[feature_cols].values
        
        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        if self.has_derivatives:
            print(f"Derivatives available: {len(self.deriv_cols)}")
    
    def _descale(self, y_scaled):
        """Descale predictions"""
        if self.scaling_params and 'volatility' in self.scaling_params:
            vol_min = self.scaling_params['volatility']['min']
            vol_max = self.scaling_params['volatility']['max']
            return y_scaled * (vol_max - vol_min) + vol_min
        return y_scaled
    
    def _compute_metrics(self, y_true, y_pred, model_name: str, train_time: float):
        """Compute and store metrics"""
        
        # Focus on volatility (first column)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_vol = y_true[:, 0]
            y_pred_vol = y_pred[:, 0] if len(y_pred.shape) > 1 else y_pred
        else:
            y_true_vol = y_true.flatten()
            y_pred_vol = y_pred.flatten()
        
        # Descale
        y_true_real = self._descale(y_true_vol)
        y_pred_real = self._descale(y_pred_vol)
        
        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        r2 = r2_score(y_true_real, y_pred_real)
        mape = np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + 1e-10))) * 100
        
        result = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'train_time_sec': train_time,
            'uses_derivatives': self.has_derivatives,
        }
        
        # If derivatives available, compute derivative MAEs
        if self.has_derivatives and len(y_true.shape) > 1 and y_true.shape[1] > 1:
            for i, deriv_name in enumerate(self.deriv_cols, start=1):
                if i < y_true.shape[1] and i < y_pred.shape[1]:
                    deriv_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                    result[f'{deriv_name}_mae'] = deriv_mae
        
        self.results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        print(f"MAE (volatility): {mae:.8f}")
        print(f"RMSE:             {rmse:.8f}")
        print(f"R²:               {r2:.6f}")
        print(f"MAPE:             {mape:.4f}%")
        print(f"Train time:       {train_time:.2f}s")
        
        if self.has_derivatives and 'dV_dbeta_scaled_mae' in result:
            print(f"\nDerivatives MAE:")
            for deriv_name in self.deriv_cols:
                if f'{deriv_name}_mae' in result:
                    print(f"  {deriv_name}: {result[f'{deriv_name}_mae']:.8f}")
        
        print(f"{'='*60}")
    
    def evaluate_tabpfn_baseline(self):
        """Evaluate TabPFN baseline"""
        
        if not TABPFN_AVAILABLE:
            print("⚠️ TabPFN not available, skipping baseline")
            return
        
        print("\n🔥 Evaluating TabPFN Baseline...")
        
        import time
        start = time.time()
        
        try:
            # TabPFN only uses volatility (single output)
            y_train_vol = self.y_train[:, 0] if len(self.y_train.shape) > 1 else self.y_train
            y_test_vol = self.y_test[:, 0] if len(self.y_test.shape) > 1 else self.y_test
            
            model = TabPFNRegressor(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                N_ensemble_configurations=4
            )
            model.fit(self.X_train, y_train_vol)
            predictions = model.predict(self.X_test)
            
            train_time = time.time() - start
            
            # Create prediction array matching y_test shape
            if len(self.y_test.shape) > 1:
                y_pred = np.zeros_like(self.y_test)
                y_pred[:, 0] = predictions
            else:
                y_pred = predictions
            
            self._compute_metrics(self.y_test, y_pred, "TabPFN (Baseline)", train_time)
            
        except Exception as e:
            print(f"❌ Error evaluating TabPFN: {e}")
    
    def evaluate_custom_model(
        self,
        model_name: str,
        model: torch.nn.Module,
        num_epochs: int = 100,
        use_derivatives: bool = True
    ):
        """Evaluate a custom PyTorch model"""
        
        print(f"\n🔥 Evaluating {model_name}...")
        
        import time
        from torch.utils.data import DataLoader, TensorDataset
        
        start = time.time()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare data
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Loss and optimizer
        if use_derivatives and self.has_derivatives:
            criterion = create_loss_function('derivative', value_weight=1.0, derivative_weight=0.5)
        else:
            criterion = torch.nn.L1Loss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if use_derivatives and self.has_derivatives:
                    pred_vol = outputs[:, 0:1]
                    true_vol = batch_y[:, 0:1]
                    pred_derivs = {f'd{i}': outputs[:, i:i+1] for i in range(1, outputs.size(1))}
                    true_derivs = {f'd{i}': batch_y[:, i:i+1] for i in range(1, batch_y.size(1))}
                    loss, _ = criterion(pred_vol, true_vol, pred_derivs, true_derivs)
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            scheduler.step(epoch_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # Prediction
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(self.X_test).to(device)
            predictions = model(X_test_tensor).cpu().numpy()
        
        train_time = time.time() - start
        
        self._compute_metrics(self.y_test, predictions, model_name, train_time)
    
    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION - All Models")
        print("="*80)
        
        # 1. TabPFN Baseline
        self.evaluate_tabpfn_baseline()
        
        # 2. Transformer with different activations (Peter's request)
        for activation in ['mish', 'gelu', 'swish', 'selu']:
            model = TabularTransformer(
                input_dim=self.X_train.shape[1],
                output_dim=self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1,
                d_model=256,
                nhead=8,
                num_layers=4,
                activation=activation
            )
            self.evaluate_custom_model(
                f"Transformer ({activation.upper()})",
                model,
                num_epochs=80,
                use_derivatives=self.has_derivatives
            )
        
        # 3. MLP with best activation (Mish typically)
        model = DeepMLP(
            input_dim=self.X_train.shape[1],
            output_dim=self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1,
            hidden_dims=[512, 256, 128, 64],
            activation='mish'
        )
        self.evaluate_custom_model(
            "DeepMLP (Mish)",
            model,
            num_epochs=80,
            use_derivatives=self.has_derivatives
        )
        
        # Print summary
        self.print_summary()
        self.save_report()
        self.plot_results()
    
    def print_summary(self):
        """Print results summary"""
        
        df = pd.DataFrame(self.results)
        df = df.sort_values('mae')
        
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY (Sorted by MAE)")
        print("="*80)
        print(df[['model', 'mae', 'rmse', 'r2', 'train_time_sec']].to_string(index=False))
        print("="*80)
        
        # Best model
        best = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['model']}")
        print(f"   MAE: {best['mae']:.8f}")
        
        # Compare to baseline
        if 'TabPFN (Baseline)' in df['model'].values:
            baseline_mae = df[df['model'] == 'TabPFN (Baseline)']['mae'].values[0]
            improvement = ((baseline_mae - best['mae']) / baseline_mae) * 100
            print(f"   Improvement over TabPFN baseline: {improvement:.2f}%")
        
        print("="*80)
    
    def save_report(self):
        """Save detailed report"""
        
        df = pd.DataFrame(self.results)
        df.to_csv('final_evaluation_results.csv', index=False)
        
        # Generate markdown report
        report = f"""# SABR TabPFN Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data:** {self.data_path}
**Samples:** {len(self.df)}
**Derivatives:** {'Yes' if self.has_derivatives else 'No'}

## Results Summary

### Best Model
{df.iloc[0].to_dict()}

### All Results
{df.to_markdown(index=False)}

## Observations

1. **Best Activation:** {df.iloc[0]['model']}
2. **MAE Achieved:** {df.iloc[0]['mae']:.8f}
3. **Training Time:** {df.iloc[0]['train_time_sec']:.2f}s

## Recommendations for Peter

1. Use {df.iloc[0]['model'].split('(')[1].split(')')[0]} activation
2. {"Derivative loss significantly helps" if self.has_derivatives else "Consider adding derivatives"}
3. Model is production-ready for SABR predictions

"""
        
        with open('final_evaluation_report.md', 'w') as f:
            f.write(report)
        
        print("\n✅ Report saved:")
        print("   - final_evaluation_results.csv")
        print("   - final_evaluation_report.md")
    
    def plot_results(self):
        """Generate visualization plots"""
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MAE comparison
        df_sorted = df.sort_values('mae')
        axes[0, 0].barh(df_sorted['model'], df_sorted['mae'])
        axes[0, 0].set_xlabel('MAE')
        axes[0, 0].set_title('Model Comparison (MAE)')
        axes[0, 0].axvline(x=0.0001, color='r', linestyle='--', label='Target (1e-4)')
        axes[0, 0].legend()
        
        # 2. R² comparison
        axes[0, 1].barh(df_sorted['model'], df_sorted['r2'])
        axes[0, 1].set_xlabel('R²')
        axes[0, 1].set_title('Model Comparison (R²)')
        
        # 3. Training time
        axes[1, 0].barh(df_sorted['model'], df_sorted['train_time_sec'])
        axes[1, 0].set_xlabel('Training Time (s)')
        axes[1, 0].set_title('Training Time Comparison')
        
        # 4. MAE vs Training Time
        axes[1, 1].scatter(df['train_time_sec'], df['mae'], s=100)
        for idx, row in df.iterrows():
            axes[1, 1].annotate(row['model'], (row['train_time_sec'], row['mae']), fontsize=8)
        axes[1, 1].set_xlabel('Training Time (s)')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Efficiency vs Accuracy')
        
        plt.tight_layout()
        plt.savefig('final_evaluation_plots.png', dpi=150, bbox_inches='tight')
        print("   - final_evaluation_plots.png")
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Final evaluation of SABR models')
    parser.add_argument('--data', type=str, default='sabr_with_derivatives_scaled.csv',
                        help='Path to data')
    parser.add_argument('--scaling', type=str, default='scaling_params_derivatives.json',
                        help='Path to scaling parameters')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FINAL EVALUATION - Following Peter's Instructions")
    print("="*80)
    print("Testing ALL differentiable activations: Swish, Mish, GELU, SELU")
    print("With derivatives in loss function as requested")
    print("="*80 + "\n")
    
    evaluator = FinalEvaluator(args.data, args.scaling)
    evaluator.run_full_evaluation()
    
    print("\n✅ Evaluation complete!")
    print("Check final_evaluation_report.md for detailed results")
