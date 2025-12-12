import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable
from .core import gated_sdpa

class GatedSDPA(nn.Module):
    def __init__(self, head_dim: int, gate_bias: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.gate_proj = nn.Linear(head_dim, head_dim, bias = gate_bias)


    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor,
                attn_mask: Optional[torch.Tensor]= None,
                is_causal: bool = False,
                dropout_p: float = 0.0,
                scale: Optional[float] = None
                ) -> torch.Tensor:
        # args: q,k,v: [batch_size, num_heads, seq_len, head_dim]

        return gated_sdpa(
            q, k, v,
            gate_w = self.gate_proj.weight,
            gate_b = self.gate_proj.bias,
            attn_mask = attn_mask,
            dropout_p = dropout_p,
            is_causal = is_causal,
            scale = scale
        )
    
class GatedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads:int, bias:bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias = bias)
        self.k_proj = nn.Linear(d_model, d_model, bias = bias)
        self.v_proj = nn.Linear(d_model, d_model, bias = bias)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)

        self.gatedSDPA = GatedSDPA(head_dim = self.head_dim)

    def forward(self,
                x: torch.Tensor,
                rotary_callback: Optional[Callable] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, # past key value for caching
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                dropout_p: float = 0.0,
                scale: Optional[float] = None
                ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        batch_size, seq_len, d_model = x.shape

        # project q, k, v 
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # apply RoPE if required
        if rotary_callback is not None:
            q, k = rotary_callback(q, k)

        # handle KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            # concat along seq_len dim
            k = torch.cat([past_k, k], dim = 2)
            v = torch.cat([past_v, v], dim = 2)

        current_kv = (k, v) # current key value
        attn_out = self.gatedSDPA(
            q, k, v,
            attn_mask = attn_mask,
            is_causal = is_causal,
            dropout_p = dropout_p,
            scale = scale
        )  # [batch_size, num_heads, seq_len, head_dim]

        # output proj
        attn_out = attn_out.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.o_proj(attn_out)

        return attn_out, current_kv

