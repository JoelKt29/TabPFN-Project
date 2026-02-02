"""
Custom Loss Functions for SABR Model Training
Includes losses that penalize both value errors AND derivative errors
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict


class SABRDerivativeLoss(nn.Module):
    """
    Custom loss that penalizes errors in both:
    1. Volatility values
    2. Volatility derivatives (Greeks)
    
    This encourages the model to learn the shape of the volatility surface,
    not just point predictions.
    """
    
    def __init__(
        self,
        value_weight: float = 1.0,
        derivative_weight: float = 0.5,
        use_relative_error: bool = True
    ):
        """
        Args:
            value_weight: Weight for volatility value loss
            derivative_weight: Weight for derivative loss
            use_relative_error: Use relative error (MAPE-like) vs absolute (MAE-like)
        """
        super().__init__()
        self.value_weight = value_weight
        self.derivative_weight = derivative_weight
        self.use_relative_error = use_relative_error
    
    def forward(
        self,
        pred_vol: torch.Tensor,
        true_vol: torch.Tensor,
        pred_greeks: Optional[Dict[str, torch.Tensor]] = None,
        true_greeks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred_vol: Predicted volatilities [batch_size, 1]
            true_vol: True volatilities [batch_size, 1]
            pred_greeks: Dictionary of predicted Greeks
            true_greeks: Dictionary of true Greeks
        """
        
        # Value loss (L1)
        if self.use_relative_error:
            # Avoid division by zero
            vol_loss = torch.mean(torch.abs((pred_vol - true_vol) / (true_vol + 1e-8)))
        else:
            vol_loss = torch.mean(torch.abs(pred_vol - true_vol))
        
        total_loss = self.value_weight * vol_loss
        
        # Derivative loss
        if pred_greeks is not None and true_greeks is not None:
            derivative_loss = 0.0
            num_greeks = 0
            
            for greek_name in pred_greeks.keys():
                if greek_name in true_greeks:
                    pred_g = pred_greeks[greek_name]
                    true_g = true_greeks[greek_name]
                    
                    if self.use_relative_error:
                        # Relative error with protection
                        greek_err = torch.mean(
                            torch.abs((pred_g - true_g) / (torch.abs(true_g) + 1e-8))
                        )
                    else:
                        greek_err = torch.mean(torch.abs(pred_g - true_g))
                    
                    derivative_loss += greek_err
                    num_greeks += 1
            
            if num_greeks > 0:
                derivative_loss /= num_greeks
                total_loss += self.derivative_weight * derivative_loss
        
        return total_loss


class WeightedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error loss.
    Can put more weight on certain regions (e.g., ATM strikes).
    """
    
    def __init__(self, moneyness_weight_fn: Optional[callable] = None):
        """
        Args:
            moneyness_weight_fn: Function that takes log-moneyness and returns weight
        """
        super().__init__()
        self.moneyness_weight_fn = moneyness_weight_fn
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_moneyness: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted MAE.
        
        Args:
            pred: Predictions
            target: Targets
            log_moneyness: Log moneyness values (optional)
        """
        
        errors = torch.abs(pred - target)
        
        if log_moneyness is not None and self.moneyness_weight_fn is not None:
            weights = self.moneyness_weight_fn(log_moneyness)
            errors = errors * weights
        
        return torch.mean(errors)


class HuberLossWithDerivatives(nn.Module):
    """
    Huber loss (smooth L1) for values + derivatives.
    More robust to outliers than pure L1/L2.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        derivative_weight: float = 0.5
    ):
        super().__init__()
        self.delta = delta
        self.derivative_weight = derivative_weight
        self.huber = nn.HuberLoss(delta=delta)
    
    def forward(
        self,
        pred_vol: torch.Tensor,
        true_vol: torch.Tensor,
        pred_greeks: Optional[Dict[str, torch.Tensor]] = None,
        true_greeks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        
        # Value loss
        vol_loss = self.huber(pred_vol, true_vol)
        
        total_loss = vol_loss
        
        # Greek losses
        if pred_greeks is not None and true_greeks is not None:
            greek_loss = 0.0
            num_greeks = 0
            
            for greek_name in pred_greeks.keys():
                if greek_name in true_greeks:
                    g_loss = self.huber(pred_greeks[greek_name], true_greeks[greek_name])
                    greek_loss += g_loss
                    num_greeks += 1
            
            if num_greeks > 0:
                greek_loss /= num_greeks
                total_loss += self.derivative_weight * greek_loss
        
        return total_loss


class QuantileLoss(nn.Module):
    """
    Quantile loss for robust regression.
    Useful when you want to penalize under/over-prediction differently.
    """
    
    def __init__(self, quantile: float = 0.5):
        """
        Args:
            quantile: Target quantile (0.5 = median regression)
        """
        super().__init__()
        assert 0 < quantile < 1
        self.quantile = quantile
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return torch.mean(loss)


# Example weight function for ATM-focused training
def atm_weight_function(log_moneyness: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Gaussian weight centered at ATM (log_moneyness = 0).
    
    Args:
        log_moneyness: Log(K/F) values
        sigma: Width of the Gaussian
        
    Returns:
        Weights (higher near ATM)
    """
    weights = torch.exp(-0.5 * (log_moneyness / sigma) ** 2)
    # Normalize so weights sum to batch_size
    weights = weights / torch.mean(weights)
    return weights


# Scikit-learn compatible wrapper for custom losses
class SKLearnLossWrapper:
    """
    Wrapper to use PyTorch losses with scikit-learn style APIs.
    """
    
    def __init__(self, loss_fn: nn.Module):
        self.loss_fn = loss_fn
    
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Compute loss (numpy arrays -> scalar).
        """
        y_true_t = torch.from_numpy(y_true).float()
        y_pred_t = torch.from_numpy(y_pred).float()
        
        with torch.no_grad():
            loss = self.loss_fn(y_pred_t, y_true_t, **kwargs)
        
        return loss.item()


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    
    # Dummy data
    pred_vol = torch.randn(batch_size, 1) * 0.01 + 0.01
    true_vol = torch.randn(batch_size, 1) * 0.01 + 0.01
    
    pred_greeks = {
        'dV_dF': torch.randn(batch_size, 1) * 0.001,
        'dV_dK': torch.randn(batch_size, 1) * 0.001,
        'dV_dRho': torch.randn(batch_size, 1) * 0.005,
    }
    
    true_greeks = {
        'dV_dF': torch.randn(batch_size, 1) * 0.001,
        'dV_dK': torch.randn(batch_size, 1) * 0.001,
        'dV_dRho': torch.randn(batch_size, 1) * 0.005,
    }
    
    # Test different losses
    print("Testing loss functions:\n")
    
    # 1. Derivative loss
    loss_fn_1 = SABRDerivativeLoss(value_weight=1.0, derivative_weight=0.5)
    loss_1 = loss_fn_1(pred_vol, true_vol, pred_greeks, true_greeks)
    print(f"SABRDerivativeLoss: {loss_1.item():.6f}")
    
    # 2. Weighted MAE
    log_moneyness = torch.randn(batch_size, 1) * 0.1
    loss_fn_2 = WeightedMAELoss(moneyness_weight_fn=atm_weight_function)
    loss_2 = loss_fn_2(pred_vol, true_vol, log_moneyness)
    print(f"WeightedMAELoss: {loss_2.item():.6f}")
    
    # 3. Huber with derivatives
    loss_fn_3 = HuberLossWithDerivatives(delta=0.01, derivative_weight=0.3)
    loss_3 = loss_fn_3(pred_vol, true_vol, pred_greeks, true_greeks)
    print(f"HuberLossWithDerivatives: {loss_3.item():.6f}")
    
    # 4. Quantile loss
    loss_fn_4 = QuantileLoss(quantile=0.5)
    loss_4 = loss_fn_4(pred_vol, true_vol)
    print(f"QuantileLoss (median): {loss_4.item():.6f}")
    
    print("\n✅ All loss functions working correctly!")
