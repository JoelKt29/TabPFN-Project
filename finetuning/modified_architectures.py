"""
Modified TabPFN Architecture with Modern Activation Functions
Implements Swish, Mish, GELU, SELU as recommended by Peter

This module provides custom transformer blocks with different activations
that can be swapped into TabPFN for experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Callable


# ============================================================================
# MODERN ACTIVATION FUNCTIONS
# ============================================================================

class Swish(nn.Module):
    """
    Swish activation: f(x) = x * sigmoid(beta * x)
    Also known as SiLU when beta=1.
    Self-gated, smooth, non-monotonic.
    """
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class Mish(nn.Module):
    """
    Mish activation: f(x) = x * tanh(softplus(x))
    Smooth, non-monotonic, self-regularizing.
    Often outperforms ReLU and Swish.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit.
    Used in BERT, GPT. Smooth approximation of ReLU.
    f(x) ≈ x * Φ(x) where Φ is CDF of standard normal.
    """
    def __init__(self, approximate: str = 'tanh'):
        """
        Args:
            approximate: 'none' for exact, 'tanh' for fast approximation
        """
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class SELU(nn.Module):
    """
    Scaled Exponential Linear Unit.
    Self-normalizing: maintains zero mean and unit variance.
    Good for deep networks without batch norm.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)


# ============================================================================
# ACTIVATION FACTORY
# ============================================================================

def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Factory function to get activation by name.
    
    Args:
        name: Activation name ('swish', 'mish', 'gelu', 'selu', 'relu', 'tanh')
        **kwargs: Additional arguments for activation
        
    Returns:
        Activation module
    """
    activations = {
        'swish': lambda: Swish(**kwargs),
        'silu': lambda: Swish(beta=1.0),  # SiLU is Swish with beta=1
        'mish': lambda: Mish(),
        'gelu': lambda: GELU(**kwargs),
        'selu': lambda: SELU(),
        'relu': lambda: nn.ReLU(),
        'leaky_relu': lambda: nn.LeakyReLU(**kwargs),
        'elu': lambda: nn.ELU(**kwargs),
        'tanh': lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
    }
    
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    
    return activations[name]()


# ============================================================================
# MODIFIED TRANSFORMER COMPONENTS
# ============================================================================

class ModifiedTransformerEncoderLayer(nn.Module):
    """
    Modified transformer encoder layer with custom activation.
    Drop-in replacement for standard TransformerEncoderLayer.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Custom activation
        self.activation = get_activation(activation)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
        """
        
        # Self-attention block
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block with custom activation
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class ModifiedMLP(nn.Module):
    """
    Multi-layer perceptron with custom activations.
    Can be used as a regression head or feature extractor.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = 'mish',
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# TABPFN-LIKE ARCHITECTURE WITH MODIFICATIONS
# ============================================================================

class CustomTabularTransformer(nn.Module):
    """
    Custom tabular transformer similar to TabPFN architecture.
    Allows experimentation with:
    - Different activation functions
    - Different depths
    - Different hidden dimensions
    - Custom regression heads
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu',
        output_dim: int = 1,
        use_mlp_head: bool = True,
        mlp_hidden_dims: Optional[list] = None,
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: FFN intermediate dimension
            dropout: Dropout rate
            activation: Activation function name
            output_dim: Number of outputs
            use_mlp_head: Use MLP regression head vs simple linear
            mlp_hidden_dims: Hidden dims for MLP head
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding (optional, can help with sequence order)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = ModifiedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # Regression head
        if use_mlp_head:
            if mlp_hidden_dims is None:
                mlp_hidden_dims = [d_model // 2, d_model // 4]
            
            self.regression_head = ModifiedMLP(
                input_dim=d_model,
                hidden_dims=mlp_hidden_dims,
                output_dim=output_dim,
                activation=activation,
                dropout=dropout,
            )
        else:
            self.regression_head = nn.Linear(d_model, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
            
        Returns:
            [batch_size, output_dim]
        """
        
        # Embed input
        x = self.input_embedding(x)  # [batch_size, d_model]
        
        # Add batch dimension for transformer (expects [batch, seq, features])
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Apply positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        
        # Take the output (squeeze sequence dimension)
        x = x.squeeze(1)  # [batch_size, d_model]
        
        # Regression head
        output = self.regression_head(x)  # [batch_size, output_dim]
        
        return output


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Not strictly necessary for tabular data, but can help.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ============================================================================
# SIMPLE FEEDFORWARD BASELINE
# ============================================================================

class DeepFeedForward(nn.Module):
    """
    Simple deep feedforward network baseline.
    Useful for comparison against transformer architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int = 1,
        activation: str = 'mish',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.network = ModifiedMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    print("Testing custom architectures...\n")
    
    batch_size = 32
    input_dim = 10
    
    # Test data
    x = torch.randn(batch_size, input_dim)
    
    # 1. Test activations
    print("1. Testing activation functions:")
    activations_to_test = ['swish', 'mish', 'gelu', 'selu', 'relu']
    
    for act_name in activations_to_test:
        act = get_activation(act_name)
        out = act(x)
        print(f"   {act_name:12s}: output shape {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")
    
    # 2. Test transformer
    print("\n2. Testing CustomTabularTransformer:")
    model = CustomTabularTransformer(
        input_dim=input_dim,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        activation='mish',
        use_mlp_head=True,
    )
    
    output = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Test feedforward baseline
    print("\n3. Testing DeepFeedForward:")
    model_ff = DeepFeedForward(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        activation='gelu',
    )
    
    output_ff = model_ff(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output_ff.shape}")
    print(f"   Number of parameters: {sum(p.numel() for p in model_ff.parameters()):,}")
    
    print("\n✅ All architectures working correctly!")
