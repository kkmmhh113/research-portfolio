"""P/C/R 후보 모델의 핵심 계산. 계산식과 설계 배경은 MODEL.md 참고."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphMessageLayer(nn.Module):
    """결합 종류별 이웃 원자 정보를 모으는 한 층."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bond_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(4)
        )
        self.output = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values, atom_mask, bond_types):
        message = self.self_projection(values)
        for bond_type, projection in enumerate(self.bond_projections, start=1):
            adjacency = (bond_types == bond_type).to(values.dtype)
            degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
            neighbors = torch.bmm(adjacency, values) / degree
            message = message + projection(neighbors)
        updated = self.norm(values + self.output(message))
        return updated * atom_mask.unsqueeze(-1).to(updated.dtype)


def dose_gate(perturbation, normalized_dose, potency_predictor, slope_raw):
    """분자 표현의 크기를 조절합니다. normalized_dose의 모양은 [batch, 1]."""
    potency = potency_predictor(perturbation)
    midpoint = torch.tanh(potency[:, :1])
    slope = F.softplus(slope_raw + 0.5 * torch.tanh(potency[:, 1:2])) + 1e-3
    gate = torch.sigmoid(slope * (normalized_dose - midpoint))
    return perturbation * gate


def condition_perturbation(effective_perturbation, biological, context_film):
    """세포와 처리 맥락으로 분자 특징을 조절하는 FiLM 형태의 계산."""
    scale, shift = context_film(biological).chunk(2, dim=-1)
    return effective_perturbation * (1.0 + 0.5 * torch.tanh(scale)) + shift


def factorized_readout(
    coefficients,
    readouts,
    response_basis,
    learned_readout_logit,
    domain_scale,
    global_bias,
    domain_bias,
):
    """gated_factorized 경로: 잠재 계수에서 유전자별 표준화 잔차로 변환."""
    learned = coefficients @ readouts.T / math.sqrt(readouts.shape[-1])
    prediction = torch.sigmoid(learned_readout_logit) * learned
    if response_basis is not None:
        prediction = prediction + coefficients @ response_basis
    return prediction * domain_scale + global_bias + domain_bias


def restore_response(standardized_residual, train_mean, train_scale, context_prior):
    """앙상블 보정 전, 표준화 잔차를 대조군 대비 발현 변화로 되돌립니다."""
    return standardized_residual * train_scale + train_mean + context_prior
