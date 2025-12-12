import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def gated_sdpa(q: torch.Tensor, 
               k: torch.Tensor, 
               v: torch.Tensor,
               gate_w: torch.Tensor,
               gate_b: Optional[torch.Tensor]= None,
               attn_mask: Optional[torch.Tensor]= None,
               dropout_p: float = 0.0,
               is_causal: bool = False,
               scale: Optional[float] = None
               ) -> torch.Tensor:
    
    """
    Calculates the Gated Scaled Dot Product Attention (Qiu et al., 2025).
    """
    # PyTorch SDPA (Flash attn handled internally)
    attn_out = F.scaled_dot_product_attention(
        q,k,v,
        attn_mask = attn_mask,
        dropout_p = dropout_p, 
        is_causal = is_causal,
        scale = scale
        )
    
    # Gate
    gate_logits = F.linear(q, gate_w, gate_b)
    gate_score = torch.sigmoid(gate_logits)

    return attn_out * gate_score
