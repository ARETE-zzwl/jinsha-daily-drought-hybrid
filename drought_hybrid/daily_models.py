from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DailyPhysicsBranch(nn.Module):
    def __init__(self, context_dim: int, reg_dim: int = 2):
        super().__init__()
        self.reg_dim = int(reg_dim)
        self.alpha_raw = nn.Parameter(torch.tensor(0.15))
        self.beta_raw = nn.Parameter(torch.tensor(0.30))
        self.gamma_raw = nn.Parameter(torch.tensor(0.45))
        self.context_adjust = nn.Linear(context_dim, self.reg_dim)
        self.cls_head = nn.Linear(context_dim + self.reg_dim, 1)

    def forward(self, pr_t: torch.Tensor, et0_t: torch.Tensor, prev_reg: torch.Tensor, context_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prev_idx30 = prev_reg[:, 0:1]
        prev_idx90 = prev_reg[:, 1:2]
        delta = self.context_adjust(context_vec)
        alpha = F.softplus(self.alpha_raw + delta[:, 0:1])
        beta = torch.sigmoid(self.beta_raw + delta[:, 1:2])
        gamma = F.softplus(self.gamma_raw + delta[:, 0:1])
        effective = pr_t - alpha * et0_t
        idx30_phy = beta * prev_idx30 + 0.15 * prev_idx90 + gamma * effective + delta[:, 0:1]
        idx90_phy = 0.7 * prev_idx90 + 0.2 * prev_idx30 + 0.1 * idx30_phy + delta[:, 1:2]
        reg_phy = torch.cat([idx30_phy, idx90_phy], dim=1)
        cls_phy = self.cls_head(torch.cat([context_vec, reg_phy], dim=1))
        return reg_phy, cls_phy


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = x.transpose(1, 2)
        y = self.conv1(y)[..., : x.size(1)]
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)[..., : x.size(1)]
        y = self.dropout(y).transpose(1, 2)
        return self.norm(residual + y)


class DailyDeepBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        station_emb_dim: int,
        static_dim: int,
        model_dim: int = 64,
        n_heads: int = 4,
        tcn_layers: int = 2,
        transformer_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim + station_emb_dim + static_dim, model_dim)
        dilations = [2**i for i in range(max(int(tcn_layers), 1))]
        self.tcn_blocks = nn.ModuleList([TemporalConvBlock(model_dim, kernel_size=3, dilation=d, dropout=dropout) for d in dilations])
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(int(transformer_layers), 1))
        self.fuse = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim), nn.GELU())
        self.reg_res_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 2))
        self.reg_abs_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 2))
        self.reg_mix_gate = nn.Sequential(nn.Linear(model_dim + 2, model_dim // 2), nn.ReLU(), nn.Linear(model_dim // 2, 2), nn.Sigmoid())
        self.cls_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, x_seq: torch.Tensor, station_emb: torch.Tensor, static_vec: torch.Tensor, prev_reg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_seq = station_emb.unsqueeze(1).repeat(1, x_seq.size(1), 1)
        static_seq = static_vec.unsqueeze(1).repeat(1, x_seq.size(1), 1)
        x = self.in_proj(torch.cat([x_seq, emb_seq, static_seq], dim=2))
        tcn_out = x
        for block in self.tcn_blocks:
            tcn_out = block(tcn_out)
        trans_out = self.transformer(x)
        h = self.fuse(torch.cat([tcn_out[:, -1, :], trans_out[:, -1, :]], dim=1))
        reg_ctx = torch.cat([h, prev_reg], dim=1)
        res_delta = 2.0 * torch.tanh(self.reg_res_head(reg_ctx))
        reg_res = prev_reg + res_delta
        reg_abs = self.reg_abs_head(reg_ctx)
        mix = self.reg_mix_gate(reg_ctx)
        reg = mix * reg_res + (1.0 - mix) * reg_abs
        cls = self.cls_head(torch.cat([h, reg], dim=1))
        return reg, cls, h


class DailyGruBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        station_emb_dim: int,
        static_dim: int,
        model_dim: int = 64,
        n_heads: int = 4,
        gru_layers: int = 1,
        transformer_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim + station_emb_dim + static_dim, model_dim)
        gru_layers = max(int(gru_layers), 1)
        self.gru = nn.GRU(model_dim, model_dim, num_layers=gru_layers, batch_first=True, dropout=(dropout if gru_layers > 1 else 0.0))
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=max(int(transformer_layers), 1))
        self.fuse = nn.Sequential(nn.Linear(model_dim * 2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim), nn.GELU())
        self.reg_res_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 2))
        self.reg_abs_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 2))
        self.reg_mix_gate = nn.Sequential(nn.Linear(model_dim + 2, model_dim // 2), nn.ReLU(), nn.Linear(model_dim // 2, 2), nn.Sigmoid())
        self.cls_head = nn.Sequential(nn.Linear(model_dim + 2, model_dim), nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, x_seq: torch.Tensor, station_emb: torch.Tensor, static_vec: torch.Tensor, prev_reg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_seq = station_emb.unsqueeze(1).repeat(1, x_seq.size(1), 1)
        static_seq = static_vec.unsqueeze(1).repeat(1, x_seq.size(1), 1)
        x = self.in_proj(torch.cat([x_seq, emb_seq, static_seq], dim=2))
        gru_out, _ = self.gru(x)
        trans_out = self.transformer(x)
        h = self.fuse(torch.cat([gru_out[:, -1, :], trans_out[:, -1, :]], dim=1))
        reg_ctx = torch.cat([h, prev_reg], dim=1)
        res_delta = 2.0 * torch.tanh(self.reg_res_head(reg_ctx))
        reg_res = prev_reg + res_delta
        reg_abs = self.reg_abs_head(reg_ctx)
        mix = self.reg_mix_gate(reg_ctx)
        reg = mix * reg_res + (1.0 - mix) * reg_abs
        cls = self.cls_head(torch.cat([h, reg], dim=1))
        return reg, cls, h


class DailyHybridModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_stations: int,
        static_dim: int,
        deep_type: str = "tcn_transformer",
        station_emb_dim: int = 8,
        model_dim: int = 64,
        n_heads: int = 4,
        tcn_layers: int = 2,
        gru_layers: int = 1,
        transformer_layers: int = 1,
        dropout: float = 0.1,
        gate_hidden_dim: int = 32,
        fusion_mode: str = "full",
        fixed_gate_value: float = 0.5,
    ):
        super().__init__()
        self.fusion_mode = str(fusion_mode).lower()
        self.fixed_gate_value = float(fixed_gate_value)
        self.station_emb_dim = int(station_emb_dim)
        self.station_embedding = (
            nn.Embedding(num_stations, station_emb_dim) if station_emb_dim > 0 else None
        )
        if deep_type == "gru_transformer":
            self.deep = DailyGruBranch(input_dim, station_emb_dim, static_dim, model_dim=model_dim, n_heads=n_heads, gru_layers=gru_layers, transformer_layers=transformer_layers, dropout=dropout)
        else:
            self.deep = DailyDeepBranch(input_dim, station_emb_dim, static_dim, model_dim=model_dim, n_heads=n_heads, tcn_layers=tcn_layers, transformer_layers=transformer_layers, dropout=dropout)
        context_dim = station_emb_dim + static_dim
        self.phy = DailyPhysicsBranch(context_dim, reg_dim=2)
        self.gate = nn.Sequential(nn.Linear(model_dim + context_dim + 5, gate_hidden_dim), nn.ReLU(), nn.Linear(gate_hidden_dim, 1), nn.Sigmoid())

    def forward(
        self,
        x_seq: torch.Tensor,
        pr_t: torch.Tensor,
        et0_t: torch.Tensor,
        prev_reg: torch.Tensor,
        station_id: torch.Tensor,
        static_vec: torch.Tensor,
    ):
        if self.station_embedding is not None:
            station_emb = self.station_embedding(station_id)
        else:
            station_emb = static_vec.new_zeros((static_vec.size(0), 0))
        context = torch.cat([station_emb, static_vec], dim=1)
        reg_dl, cls_dl, h = self.deep(x_seq, station_emb, static_vec, prev_reg)
        reg_phy, cls_phy = self.phy(pr_t, et0_t, prev_reg, context)
        if self.fusion_mode == "dl_only":
            g = torch.ones((x_seq.size(0), 1), dtype=x_seq.dtype, device=x_seq.device)
            reg = reg_dl
            cls = cls_dl
        elif self.fusion_mode == "phy_only":
            g = torch.zeros((x_seq.size(0), 1), dtype=x_seq.dtype, device=x_seq.device)
            reg = reg_phy
            cls = cls_phy
        elif self.fusion_mode == "fixed_gate":
            g = torch.full(
                (x_seq.size(0), 1),
                self.fixed_gate_value,
                dtype=x_seq.dtype,
                device=x_seq.device,
            )
            reg = g * reg_dl + (1.0 - g) * reg_phy
            cls = g * cls_dl + (1.0 - g) * cls_phy
        else:
            g = self.gate(torch.cat([h, context, reg_dl, reg_phy, cls_dl], dim=1))
            reg = g * reg_dl + (1.0 - g) * reg_phy
            cls = g * cls_dl + (1.0 - g) * cls_phy
        return reg, cls, reg_dl, reg_phy, g

