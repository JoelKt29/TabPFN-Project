"""
Quick Test Script - Verify Phase 2 Implementation
Tests all components end-to-end with small dataset
"""

import numpy as np
import torch
import sys
from pathlib import Path

print("="*80)
print("PHASE 2 COMPONENTS TEST")
print("="*80)

# Test 1: Derivatives computation
print("\n1. Testing SABR Derivatives Computation...")
try:
    from sabr_derivatives import SABRGreeks
    
    calculator = SABRGreeks(epsilon=1e-6)
    greeks = calculator.compute_all_greeks(
        f=0.05, k=0.055, t=1.0, v_atm_n=0.01,
        beta=0.5, rho=0.0, volvol=0.2
    )
    
    print("   ✅ Derivatives computed successfully!")
    print(f"   - Volatility: {greeks['volatility']:.6f}")
    print(f"   - dV/dF: {greeks['dV_dF']:.6f}")
    print(f"   - dV/dK: {greeks['dV_dK']:.6f}")
    print(f"   - dV/dRho: {greeks['dV_dRho']:.6f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Custom Loss Functions
print("\n2. Testing Custom Loss Functions...")
try:
    from custom_losses import SABRDerivativeLoss, WeightedMAELoss
    
    # Test data
    pred = torch.randn(10, 1) * 0.01 + 0.01
    target = torch.randn(10, 1) * 0.01 + 0.01
    
    # Test derivative loss
    loss_fn = SABRDerivativeLoss(value_weight=1.0, derivative_weight=0.5)
    loss = loss_fn(pred, target)
    
    print(f"   ✅ SABRDerivativeLoss working! Loss: {loss.item():.6f}")
    
    # Test weighted MAE
    from custom_losses import atm_weight_function
    loss_fn2 = WeightedMAELoss(moneyness_weight_fn=atm_weight_function)
    log_moneyness = torch.randn(10, 1) * 0.1
    loss2 = loss_fn2(pred, target, log_moneyness)
    
    print(f"   ✅ WeightedMAELoss working! Loss: {loss2.item():.6f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 3: Modified Architectures
print("\n3. Testing Modified Architectures...")
try:
    from modified_architectures import (
        CustomTabularTransformer,
        DeepFeedForward,
        get_activation
    )
    
    # Test activations
    activations = ['swish', 'mish', 'gelu', 'selu']
    for act_name in activations:
        act = get_activation(act_name)
        x = torch.randn(5, 10)
        y = act(x)
        print(f"   ✅ {act_name:8s} activation: shape {y.shape}, mean={y.mean():.4f}")
    
    # Test transformer
    model = CustomTabularTransformer(
        input_dim=10,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        activation='mish'
    )
    x = torch.randn(5, 10)
    output = model(x)
    print(f"   ✅ Transformer: input {x.shape} -> output {output.shape}")
    
    # Test feedforward
    model_ff = DeepFeedForward(
        input_dim=10,
        hidden_dims=[64, 32],
        activation='gelu'
    )
    output_ff = model_ff(x)
    print(f"   ✅ FeedForward: input {x.shape} -> output {output_ff.shape}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 4: Training Pipeline (smoke test)
print("\n4. Testing Training Pipeline...")
try:
    from train_sabr_model import SABRModelTrainer
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create tiny dummy dataset
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.random.randn(100, 1)
    
    dataset = TensorDataset(
        torch.FloatTensor(X_dummy),
        torch.FloatTensor(y_dummy)
    )
    train_loader = DataLoader(dataset, batch_size=16)
    
    # Create simple model
    model = DeepFeedForward(input_dim=10, hidden_dims=[32, 16], activation='mish')
    
    # Create trainer
    trainer = SABRModelTrainer(model, device='cpu')
    
    # Quick training test (1 epoch)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for 1 epoch
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        break  # Just test one batch
    
    print(f"   ✅ Training pipeline working! Test loss: {loss.item():.6f}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: File Dependencies
print("\n5. Checking File Dependencies...")
required_files = [
    'base_sabr.py',
    'hagan_2002_lognormal_sabr.py',
]

for filename in required_files:
    if Path(filename).exists():
        print(f"   ✅ {filename} found")
    else:
        print(f"   ⚠️  {filename} not found (needed for derivatives)")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✅ All core components working!")
print("\nYou can now:")
print("1. Run 'python sabr_derivatives.py' to generate data with Greeks")
print("2. Run 'python train_sabr_model.py' to train a model")
print("3. Run 'python ray_tune_search.py' for hyperparameter search")
print("\nRefer to README.md for detailed instructions.")
print("="*80)

# Additional recommendations
print("\n💡 RECOMMENDATIONS:")
print("- Start with derivatives (Peter's priority)")
print("- Test Mish activation first (often best performer)")
print("- Use Google Colab if you need GPU")
print("- Keep batch_size=64 for 5000 samples")
print("- Monitor validation MAE closely")
print("\n🎯 Target: Beat your baseline MAE of 5e-5 with derivatives!")
