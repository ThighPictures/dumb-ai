
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin
from models.sparse_embedding import CastedSparseEmbedding
from models.layers import CastedEmbedding, CastedLinear



class GriffinBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(hidden_size=dim, num_heads=heads, causal=False)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLU(dim, expansion=4.0)
        self.gate = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, h: torch.Tensor, inject: Optional[torch.Tensor] = None):
        x = self.norm1(h)
        if inject is not None:
            x = x + inject
        x = rms_norm(x + self.attn(hidden_states=x), 1e-5)
        x = rms_norm(x + self.gate.tanh() * self.mlp(self.norm2(x)), 1e-5)
        return x



@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 10000
    vocab_size: int = 32768

    # Old cycle/layer params are ignored — kept for compatibility
    H_cycles: int = 8
    L_cycles: int = 12
    H_layers: int = 12
    L_layers: int = 16

    hidden_size: int = 1024
    expansion: float = 4.0
    num_heads: int = 16
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    halt_max_steps: int = 64
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"


# ===================================================================
# Tiered Fast Proposers (32 tiny -> 8 medium -> 3 heavy)
# ===================================================================
class TieredProposer(nn.Module):
    def __init__(self, dim, layers, count):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(*[
                GriffinBlock(dim, heads=max(8, dim//64)) for _ in range(layers)
            ]) for _ in range(count)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, states, inject):
        outs = []
        for i, net in enumerate(self.nets):
            h = net(states[i] if i < states.shape[0] else states[0])
            outs.append(self.norm(h + inject))
        return torch.stack(outs)



class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.forward_dtype)
        d = config.hidden_size

        # Shared embedding
        self.embed = CastedEmbedding(config.vocab_size, d, cast_to=self.dtype)
        self.rotary = RotaryEmbedding(dim=d // config.num_heads, base=config.rope_theta)

        # Tiered proposers
        self.tier1 = TieredProposer(384, 2, 32)   # 32 × ~22M
        self.tier2 = TieredProposer(768, 4, 8)    # 8 × ~78M
        self.tier3 = TieredProposer(1024, 8, 3)   # 3 × ~180M

        # Up-projectors
        self.up1 = nn.Linear(384, 768, bias=False)
        self.up2 = nn.Linear(768, 1024, bias=False)

        # Deep verifiers with fast/slow critics
        self.verifiers = nn.ModuleList([
            nn.ModuleList([GriffinBlock(1024) for _ in range(32)]) for _ in range(2)
        ])
        self.fast_critic = nn.Linear(1024, 1)
        self.slow_critic = nn.Linear(1024, 1)


        self.lm_head = CastedLinear(d, config.vocab_size, bias=False)
        self.q_head = CastedLinear(d, 2, bias=True)

        self.apply(self._init)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    # Fake input embedding to keep old code happy
    def _input_embeddings(self, x, pid):
        return self.embed(x.to(torch.int32)) * math.sqrt(self.config.hidden_size)

    def forward(self, carry, batch):
        # This is now the full MurderTree step
        inp = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        cos_sin = self.rotary()

        z_H, z_L = carry.z_H, carry.z_L

        # === Hierarchical proposing ===
        t1_out = self.tier1(z_H[:32], inp + z_L)
        t2_in = self.up1(t1_out.mean(0, keepdim=True))
        t2_out = self.tier2(z_H[32:40], t2_in + inp)
        t3_in = self.up2(t2_out.mean(0, keepdim=True))
        candidates = self.tier3(z_H[40:43], t3_in + inp)

        # === Deep verification (fast + occasional slow) ===
        slow = (carry.steps[0] % 8 == 0)
        scores = self.fast_critic(candidates.mean(0))
        if slow:
            for block in self.verifiers[0]:
                candidates = block(candidates)
            scores = self.slow_critic(candidates.mean(0))

        # Update carries (fake hierarchy for API compatibility)
        new_z_H = candidates.mean(0) + z_H
        new_z_L = inp + z_L

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=new_z_H.detach(), z_L=new_z_L.detach()
        )

        logits = self.lm_head(new_z_H)[:, :batch["inputs"].shape[1]]
        q = self.q_head(new_z_H[:, 0])

        return new_carry, logits, (q[..., 0], q[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return None  # kept for compatibility

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        bs = batch["inputs"].shape[0]
        d = self.config.hidden_size
        dtype = getattr(torch, self.config.forward_dtype)
        device = next(self.inner.parameters()).device

        fake_H = torch.zeros(43, bs, d, device=device, dtype=dtype)  # 32+8+3 tiers
        fake_L = torch.zeros(bs, self.config.seq_len + 64, d, device=device, dtype=dtype)

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=HierarchicalReasoningModel_ACTV1InnerCarry(z_H=fake_H, z_L=fake_L),
            steps=torch.zeros(bs, dtype=torch.int32, device=device),
            halted=torch.zeros(bs, dtype=torch.bool, device=device),
            current_data={k: v.clone() for k, v in batch.items()}
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner, logits, (q_halt, q_cont) = self.inner(carry.inner_carry, batch)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt,
            "q_continue_logits": q_cont,
        }

        with torch.no_grad():
            carry.steps += 1
            halted = carry.steps >= self.config.halt_max_steps
            carry.halted = halted
            carry.inner_carry = new_inner

            if self.training:
                halted = halted | (q_halt > q_cont)

        return carry, outputs


        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
