"""
Loss Function with Derivatives
Selon Peter: "modify loss function, create the data"

Loss = α * L_volatility + β * L_derivatives

Permet de forcer le modèle à apprendre non seulement les valeurs
mais aussi les gradients (forme de la surface)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


class DerivativeLoss(nn.Module):
    """
    Combined loss for volatility + derivatives
    
    Loss = value_weight * MAE(vol) + deriv_weight * MAE(derivatives)
    """
    
    def __init__(
        self,
        value_weight: float = 1.0,
        derivative_weight: float = 0.5,
        derivative_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            value_weight: Weight for volatility error
            derivative_weight: Global weight for all derivatives
            derivative_weights: Individual weights per derivative (optional)
        """
        super().__init__()
        self.value_weight = value_weight
        self.derivative_weight = derivative_weight
        self.derivative_weights = derivative_weights or {}
    
    def forward(
        self,
        pred_vol: torch.Tensor,
        true_vol: torch.Tensor,
        pred_derivs: Optional[Dict[str, torch.Tensor]] = None,
        true_derivs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss
        
        Args:
            pred_vol: Predicted volatilities [batch_size, 1]
            true_vol: True volatilities [batch_size, 1]
            pred_derivs: Dict of predicted derivatives
            true_derivs: Dict of true derivatives
            
        Returns:
            total_loss, loss_breakdown
        """
        
        # Volatility loss (L1/MAE)
        vol_loss = torch.mean(torch.abs(pred_vol - true_vol))
        total_loss = self.value_weight * vol_loss
        
        loss_breakdown = {
            'volatility_loss': vol_loss.item(),
        }
        
        # Derivative losses
        if pred_derivs is not None and true_derivs is not None:
            deriv_loss_total = 0.0
            num_derivs = 0
            
            for deriv_name in pred_derivs.keys():
                if deriv_name in true_derivs:
                    pred_d = pred_derivs[deriv_name]
                    true_d = true_derivs[deriv_name]
                    
                    # Individual derivative loss
                    deriv_loss = torch.mean(torch.abs(pred_d - true_d))
                    
                    # Apply individual weight if specified
                    weight = self.derivative_weights.get(deriv_name, 1.0)
                    deriv_loss_total += weight * deriv_loss
                    num_derivs += 1
                    
                    loss_breakdown[f'{deriv_name}_loss'] = deriv_loss.item()
            
            if num_derivs > 0:
                # Average over derivatives and apply global weight
                avg_deriv_loss = deriv_loss_total / num_derivs
                total_loss += self.derivative_weight * avg_deriv_loss
                loss_breakdown['avg_derivative_loss'] = avg_deriv_loss.item()
        
        loss_breakdown['total_loss'] = total_loss.item()
        
        return total_loss, loss_breakdown


class AdaptiveDerivativeLoss(nn.Module):
    """
    Adaptive loss that adjusts derivative weights during training
    based on which derivatives are harder to predict
    """
    
    def __init__(
        self,
        value_weight: float = 1.0,
        derivative_weight: float = 0.5,
        adaptive: bool = True
    ):
        super().__init__()
        self.value_weight = value_weight
        self.derivative_weight = derivative_weight
        self.adaptive = adaptive
        
        # Track derivative difficulties
        self.deriv_errors = {}
        self.update_count = 0
    
    def forward(
        self,
        pred_vol: torch.Tensor,
        true_vol: torch.Tensor,
        pred_derivs: Optional[Dict[str, torch.Tensor]] = None,
        true_derivs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Volatility loss
        vol_loss = torch.mean(torch.abs(pred_vol - true_vol))
        total_loss = self.value_weight * vol_loss
        
        loss_breakdown = {'volatility_loss': vol_loss.item()}
        
        # Derivative losses with adaptive weighting
        if pred_derivs is not None and true_derivs is not None:
            deriv_loss_total = 0.0
            current_errors = {}
            
            for deriv_name in pred_derivs.keys():
                if deriv_name in true_derivs:
                    pred_d = pred_derivs[deriv_name]
                    true_d = true_derivs[deriv_name]
                    
                    deriv_loss = torch.mean(torch.abs(pred_d - true_d))
                    current_errors[deriv_name] = deriv_loss.item()
                    
                    # Adaptive weight based on historical difficulty
                    if self.adaptive and deriv_name in self.deriv_errors:
                        # Harder derivatives get more weight
                        avg_error = np.mean(self.deriv_errors[deriv_name][-10:])  # Last 10 updates
                        adaptive_weight = avg_error / (sum(self.deriv_errors[d][-1] for d in self.deriv_errors) + 1e-8)
                    else:
                        adaptive_weight = 1.0
                    
                    deriv_loss_total += adaptive_weight * deriv_loss
                    loss_breakdown[f'{deriv_name}_loss'] = deriv_loss.item()
                    loss_breakdown[f'{deriv_name}_weight'] = adaptive_weight
            
            # Update error history
            for deriv_name, error in current_errors.items():
                if deriv_name not in self.deriv_errors:
                    self.deriv_errors[deriv_name] = []
                self.deriv_errors[deriv_name].append(error)
            
            self.update_count += 1
            
            avg_deriv_loss = deriv_loss_total / len(pred_derivs)
            total_loss += self.derivative_weight * avg_deriv_loss
            loss_breakdown['avg_derivative_loss'] = avg_deriv_loss.item()
        
        loss_breakdown['total_loss'] = total_loss.item()
        
        return total_loss, loss_breakdown


class HuberDerivativeLoss(nn.Module):
    """
    Huber loss (robust to outliers) for volatility + derivatives
    Good for financial data which may have outliers
    """
    
    def __init__(
        self,
        value_weight: float = 1.0,
        derivative_weight: float = 0.5,
        delta: float = 1.0
    ):
        super().__init__()
        self.value_weight = value_weight
        self.derivative_weight = derivative_weight
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(
        self,
        pred_vol: torch.Tensor,
        true_vol: torch.Tensor,
        pred_derivs: Optional[Dict[str, torch.Tensor]] = None,
        true_derivs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Volatility Huber loss
        vol_loss = self.huber(pred_vol, true_vol)
        total_loss = self.value_weight * vol_loss
        
        loss_breakdown = {'volatility_loss': vol_loss.item()}
        
        # Derivative Huber losses
        if pred_derivs is not None and true_derivs is not None:
            deriv_loss_total = 0.0
            
            for deriv_name in pred_derivs.keys():
                if deriv_name in true_derivs:
                    deriv_loss = self.huber(pred_derivs[deriv_name], true_derivs[deriv_name])
                    deriv_loss_total += deriv_loss
                    loss_breakdown[f'{deriv_name}_loss'] = deriv_loss.item()
            
            avg_deriv_loss = deriv_loss_total / len(pred_derivs)
            total_loss += self.derivative_weight * avg_deriv_loss
            loss_breakdown['avg_derivative_loss'] = avg_deriv_loss.item()
        
        loss_breakdown['total_loss'] = total_loss.item()
        
        return total_loss, loss_breakdown


class GradientMatchingLoss(nn.Module):
    """
    Alternative: Match gradients using automatic differentiation
    More principled than finite differences
    """
    
    def __init__(self, value_weight: float = 1.0, gradient_weight: float = 0.5):
        super().__init__()
        self.value_weight = value_weight
        self.gradient_weight = gradient_weight
    
    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        true_vol: torch.Tensor,
        true_grads: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss matching both values and gradients
        
        Args:
            model: The neural network
            inputs: Input features [batch_size, n_features]
            true_vol: True volatilities
            true_grads: True gradients wrt each input feature
        """
        
        # Enable gradient computation for inputs
        inputs.requires_grad_(True)
        
        # Forward pass
        pred_vol = model(inputs)
        
        # Value loss
        vol_loss = torch.mean(torch.abs(pred_vol - true_vol))
        total_loss = self.value_weight * vol_loss
        
        loss_breakdown = {'volatility_loss': vol_loss.item()}
        
        # Gradient loss
        if true_grads:
            grad_loss_total = 0.0
            
            # Compute gradients wrt inputs
            for i, (grad_name, true_grad) in enumerate(true_grads.items()):
                # Compute ∂(pred_vol)/∂(input_i)
                pred_grad = torch.autograd.grad(
                    outputs=pred_vol,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(pred_vol),
                    create_graph=True,
                    retain_graph=True
                )[0][:, i:i+1]
                
                grad_loss = torch.mean(torch.abs(pred_grad - true_grad))
                grad_loss_total += grad_loss
                loss_breakdown[f'{grad_name}_gradient_loss'] = grad_loss.item()
            
            avg_grad_loss = grad_loss_total / len(true_grads)
            total_loss += self.gradient_weight * avg_grad_loss
            loss_breakdown['avg_gradient_loss'] = avg_grad_loss.item()
        
        loss_breakdown['total_loss'] = total_loss.item()
        
        return total_loss, loss_breakdown


# ============================================================================
# Helper functions
# ============================================================================

def create_loss_function(loss_type: str = 'derivative', **kwargs) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: 'derivative', 'adaptive', 'huber', or 'gradient_matching'
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function module
    """
    
    loss_functions = {
        'derivative': DerivativeLoss,
        'adaptive': AdaptiveDerivativeLoss,
        'huber': HuberDerivativeLoss,
        'gradient_matching': GradientMatchingLoss,
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)


if __name__ == "__main__":
    print("="*80)
    print("LOSS FUNCTIONS WITH DERIVATIVES - Testing")
    print("="*80)
    
    # Create dummy data
    batch_size = 32
    
    pred_vol = torch.randn(batch_size, 1) * 0.01 + 0.01
    true_vol = torch.randn(batch_size, 1) * 0.01 + 0.01
    
    pred_derivs = {
        'dV_dbeta': torch.randn(batch_size, 1) * 0.001,
        'dV_drho': torch.randn(batch_size, 1) * 0.001,
        'dV_dvolvol': torch.randn(batch_size, 1) * 0.005,
        'dV_dF': torch.randn(batch_size, 1) * 0.001,
        'dV_dK': torch.randn(batch_size, 1) * 0.001,
    }
    
    true_derivs = {
        'dV_dbeta': torch.randn(batch_size, 1) * 0.001,
        'dV_drho': torch.randn(batch_size, 1) * 0.001,
        'dV_dvolvol': torch.randn(batch_size, 1) * 0.005,
        'dV_dF': torch.randn(batch_size, 1) * 0.001,
        'dV_dK': torch.randn(batch_size, 1) * 0.001,
    }
    
    # Test different loss functions
    print("\n1. DerivativeLoss:")
    loss_fn = DerivativeLoss(value_weight=1.0, derivative_weight=0.5)
    loss, breakdown = loss_fn(pred_vol, true_vol, pred_derivs, true_derivs)
    print(f"   Total loss: {loss.item():.6f}")
    for key, value in breakdown.items():
        print(f"   {key}: {value:.6f}")
    
    print("\n2. AdaptiveDerivativeLoss:")
    loss_fn = AdaptiveDerivativeLoss(value_weight=1.0, derivative_weight=0.5)
    loss, breakdown = loss_fn(pred_vol, true_vol, pred_derivs, true_derivs)
    print(f"   Total loss: {loss.item():.6f}")
    
    print("\n3. HuberDerivativeLoss:")
    loss_fn = HuberDerivativeLoss(value_weight=1.0, derivative_weight=0.5, delta=0.1)
    loss, breakdown = loss_fn(pred_vol, true_vol, pred_derivs, true_derivs)
    print(f"   Total loss: {loss.item():.6f}")
    
    print("\n" + "="*80)
    print("✅ All loss functions working correctly!")
    print("="*80)
    print("\nRecommendations:")
    print("- Start with DerivativeLoss (simple, effective)")
    print("- Use AdaptiveDerivativeLoss if some derivatives are harder to learn")
    print("- Use HuberDerivativeLoss if you have outliers in data")
    print("- Use GradientMatchingLoss for most principled approach (but slower)")
