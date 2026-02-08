import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class DerivativeLoss(nn.Module):
    """
    Standard Sobolev loss comparing volatility values and their derivatives.
    """
    def __init__(self, value_weight: float = 1.0, derivative_weight: float = 0.5):
        super().__init__()
        self.value_weight = value_weight
        self.derivative_weight = derivative_weight
    
    def forward(
        self, 
        pred_vol: torch.Tensor, 
        true_vol: torch.Tensor, 
        pred_derivs: Optional[Dict[str, torch.Tensor]] = None, 
        true_derivs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Compute Mean Absolute Error for volatility
        vol_loss = torch.mean(torch.abs(pred_vol - true_vol))
        total_loss = vol_loss * self.value_weight
        
        loss_breakdown = {'volatility_loss': vol_loss.item()}
        
        # Compute Mean Absolute Error for derivatives (Greeks)
        if pred_derivs is not None and true_derivs is not None:
            deriv_loss_total = 0.0
            num_derivs = 0
            
            for name, p_deriv in pred_derivs.items():
                if name in true_derivs:
                    t_deriv = true_derivs[name]
                    d_loss = torch.mean(torch.abs(p_deriv - t_deriv))
                    deriv_loss_total += d_loss
                    num_derivs += 1
                    loss_breakdown[f'{name}_loss'] = d_loss.item()
            
            if num_derivs > 0:
                avg_deriv_loss = deriv_loss_total / num_derivs
                total_loss += self.derivative_weight * avg_deriv_loss
                loss_breakdown['avg_derivative_loss'] = avg_deriv_loss.item()
        
        loss_breakdown['total_loss'] = total_loss.item()
        return total_loss, loss_breakdown

if __name__ == "__main__":
    # Unit Test with synthetic data
    batch_size = 64
    
    # Simulated  outputs
    pred_v = torch.randn(batch_size, 1) * 0.01 + 0.01
    true_v = torch.randn(batch_size, 1) * 0.01 + 0.01
    greeks = ['dV_dbeta', 'dV_drho', 'dV_dvolvol', 'dV_dF', 'dV_dK']
    pred_d = {g: torch.randn(batch_size, 1) * 0.001 for g in greeks}
    true_d = {g: torch.randn(batch_size, 1) * 0.001 for g in greeks}
    
    criterion = DerivativeLoss(value_weight=1.0, derivative_weight=0.5)
    loss, breakdown = criterion(pred_v, true_v, pred_d, true_d)
    
    print("\n--- Loss Test Results ---")
    for key, value in breakdown.items():
        print(f"{key:20}: {value:.2f}")