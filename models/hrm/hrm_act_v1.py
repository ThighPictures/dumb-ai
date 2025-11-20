import math
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding
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


class TieredProposer(nn.Module):
    def __init__(self, dim, layers, count):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(*[GriffinBlock(dim, heads=max(8, dim // 64)) for _ in range(layers)])
            for _ in range(count)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, states, inject):
        # states: (count, bs, dim) or smaller if fewer than count
        count = len(self.nets)
        bs = inject.shape[0]
        outs = []
        for i, net in enumerate(self.nets):
            s = states[min(i, states.shape[0] - 1)]  # repeat last if not enough
            h = net(s + inject)
            outs.append(self.norm(h))
        return torch.stack(outs, dim=0)  # (count, bs, dim)


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int = 32768
    hidden_size: int = 1024
    num_heads: int = 16
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    halt_max_steps: int = 64
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "bfloat16"
    ponder_weight: float = 0.01  # λ for ponder loss


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor           # (43, bs, d) accumulated high-level reasoning states
    z_L: torch.Tensor           # (bs, seq_len+pad, d) accumulated input features
    accumulated_logits: torch.Tensor   # (bs, seq_len, vocab) weighted sum so far
    accumulated_p_cont: torch.Tensor   # (bs,) remaining probability mass
    total_p_halt: torch.Tensor         # (bs,) total halting probability spent


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor         # (bs,) int32
    halted: torch.Tensor        # (bs,) bool
    current_data: Dict[str, torch.Tensor]



class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config):
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.forward_dtype)
        d = config.hidden_size

        self.embed = CastedEmbedding(config.vocab_size, d, cast_to=self.dtype)
        self.rotary = RotaryEmbedding(dim=d // config.num_heads, base=config.rope_theta)


        self.tier1 = TieredProposer(384, 2, 32)   # 32 tiny
        self.tier2 = TieredProposer(768, 4, 8)    # 8 medium
        self.tier3 = TieredProposer(1024, 8, 3)   # 3 heavy

        self.up1 = nn.Linear(384, 768, bias=False)
        self.up2 = nn.Linear(768, 1024, bias=False)


        self.gate_t1 = nn.Linear(384, 1)
        self.gate_t2 = nn.Linear(768, 1)
        self.gate_t3 = nn.Linear(1024, 1)
        self.fusion_gate = nn.Sequential(nn.Linear(1024, 1024), nn.SiLU(), nn.Linear(1024, 1))

        # deep verifiers (fast + slow path)
        self.verifiers_fast = nn.ModuleList([GriffinBlock(1024) for _ in range(16)])
        self.verifiers_slow = nn.ModuleList([GriffinBlock(1024) for _ in range(32)])

        self.fast_critic = nn.Linear(1024, 1024)
        self.slow_critic = nn.Linear(1024, 1024)


        self.lm_head = CastedLinear(d, config.vocab_size, bias=False)
        self.q_head = CastedLinear(d, 2, bias=True)  # [halt, continue]

        self.apply(self._init_weights)
        with torch.no_grad():
            self.q_head.bias.fill_(-5.0)
            self.q_head.weight.zero_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _input_embeddings(self, x):
        return self.embed(x.to(torch.int32)) * math.sqrt(self.config.hidden_size)

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, torch.Tensor]:
        inp = self._input_embeddings(batch["inputs"])  # (bs, seq, d)
        z_H, z_L = carry.z_H, carry.z_L
        bs, seq_len, d = inp.shape

        #tiered hierarchical proposing with learned gating
        t1_out = self.tier1(z_H[:32], inp + z_L[:, :seq_len])          # (32, bs, 384)
        t1_g = torch.sigmoid(self.gate_t1(t1_out.mean(2)))             # (bs, 1)
        t1_weighted = (t1_out * t1_g.unsqueeze(1)).sum(0)               # (bs, 384)

        t2_in = self.up1(t1_weighted) + inp
        t2_out = self.tier2(z_H[32:40], t2_in)                         # (8, bs, 768)
        t2_g = torch.sigmoid(self.gate_t2(t2_out.mean(2)))
        t2_weighted = (t2_out * t2_g.unsqueeze(1)).sum(0)

        t3_in = self.up2(t2_weighted) + inp
        t3_out = self.tier3(z_H[40:43], t3_in)                         # (3, bs, 1024)
        t3_g = torch.sigmoid(self.gate_t3(t3_out.mean(2)))
        candidate = (t3_out * t3_g.unsqueeze(1)).sum(0)                # (bs, 1024)


        fusion_g = torch.sigmoid(self.fusion_gate(candidate))          # (bs, 1)
        proposal = candidate * fusion_g                                # gated new reasoning

        # Deep verification (slow every 8 steps)
        verify_input = proposal.unsqueeze(0)
        slow = (carry.steps[0] % 8 == 0) if hasattr(carry, "steps") else False
        if slow:
            for blk in self.verifiers_slow:
                verify_input = blk(verify_input)
            verified = self.slow_critic(verify_input.squeeze(0))
        else:
            for blk in self.verifiers_fast:
                verify_input = blk(verify_input)
            verified = self.fast_critic(verify_input.squeeze(0))

        h_step = verified + z_H.mean(0)


        q_logits = self.q_head(h_step)                     # (bs, 2)
        p_halt = torch.sigmoid(q_logits[:, 0])
        p_continue_step = torch.sigmoid(q_logits[:, 1])

        step_logits = self.lm_head(h_step)[:, :seq_len]


        new_z_H = z_H + proposal.unsqueeze(0)
        new_z_L = z_L + F.pad(inp, (0, 0, 0, 64))[:z_L.shape[1]]

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=new_z_H.detach(),
            z_L=new_z_L.detach(),
            accumulated_logits=carry.accumulated_logits,
            accumulated_p_cont=carry.accumulated_p_cont,
            total_p_halt=carry.total_p_halt,
        )

        return new_carry, step_logits, q_logits


class HierarchicalReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        bs = batch["inputs"].shape[0]
        d = self.config.hidden_size
        dtype = getattr(torch, self.config.forward_dtype)
        device = next(self.inner.parameters()).device
        seq_len = batch["inputs"].shape[1]

        fake_H = torch.zeros(43, bs, d, device=device, dtype=dtype)
        fake_L = torch.zeros(bs, seq_len + 64, d, device=device, dtype=dtype)

        inner = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=fake_H,
            z_L=fake_L,
            accumulated_logits=torch.zeros(bs, seq_len, self.config.vocab_size, device=device, dtype=dtype),
            accumulated_p_cont=torch.ones(bs, device=device, dtype=dtype),
            total_p_halt=torch.zeros(bs, device=device, dtype=dtype),
        )

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner,
            steps=torch.zeros(bs, dtype=torch.int32, device=device),
            halted=torch.zeros(bs, dtype=torch.bool, device=device),
            current_data={k: v.clone() for k, v in batch.items()},
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        inner = carry.inner_carry
        new_inner, step_logits, q_logits = self.inner(inner, batch)

        q_halt_raw, q_cont_raw = q_logits[:, 0], q_logits[:, 1]
        p_halt = torch.sigmoid(q_halt_raw)
        p_cont_step = torch.sigmoid(q_cont_raw)

        remaining_mass = inner.accumulated_p_cont
        p_cont_this_step = remaining_mass * p_cont_step

        weighted_logits = step_logits * p_cont_this_step.unsqueeze(-1).unsqueeze(-1)

        new_accum_logits = inner.accumulated_logits + weighted_logits
        new_total_p_halt = inner.total_p_halt + p_halt * remaining_mass
        new_remaining = remaining_mass * (1.0 - p_halt)

        final_logits = new_accum_logits.clone()
        remainder = (1.0 - new_total_p_halt - new_remaining).unsqueeze(-1).unsqueeze(-1)
        final_logits = final_logits + step_logits * remainder.clamp(min=0.0)

        new_inner.accumulated_logits = new_accum_logits.detach()
        new_inner.accumulated_p_cont = new_remaining.detach()
        new_inner.total_p_halt = new_total_p_halt.detach()

        with torch.no_grad():
            carry.steps += 1
            halt_now = (carry.steps >= 2) & (p_halt >= 0.99)
            if self.training:
                explore = torch.rand_like(p_halt) < self.config.halt_exploration_prob
                halt_now = halt_now | explore
            carry.halted = carry.halted | halt_now | (carry.steps >= self.config.halt_max_steps)

        carry.inner_carry = new_inner

        outputs = {
            "logits": final_logits,
            "q_halt_logits": q_halt_raw,
            "q_continue_logits": q_cont_raw,
            "halting_probability": new_total_p_halt + p_halt * remaining_mass,
            "steps_taken": carry.steps.float(),
        }


        if self.training:
            ponder_cost = carry.steps.float() + (1.0 - new_total_p_halt - new_remaining).abs()
            outputs["ponder_loss"] = ponder_cost.mean() * self.config.ponder_weight

        return carry, outputs
