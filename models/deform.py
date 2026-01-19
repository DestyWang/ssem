from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .utils import (
    ensure_4d,
    gradient_l2,
    normalize_distribution,
    sample_field,
    smooth_field,
    warp_image,
)
from .wasserstein2_loss import Wasserstein2Loss

Tensor = torch.Tensor


class LDDMM(nn.Module):
    """
    基于最优传输的 2D LDDMM（支持 GPU）。

    该实现使用速度场 v_t 的欧拉积分来生成位移场，并在配准损失中
    使用 Wasserstein barycenter 目标：对当前切片与相邻切片的 W2
    距离加权求和。
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        time_steps: int = 5,
        smooth_sigma: float = 1.0,
        reg_weight: float = 1e-3,
        sum_target: float = 1e7,
        device: torch.device = torch.device('cuda'),
        w2_paras: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        参数
        ----
        image_shape : Tuple[int, int]
            图像形状 (H, W)。
        time_steps : int
            时间离散步数 T。
        smooth_sigma : float
            高斯平滑标准差。
        reg_weight : float
            正则化权重。
        w2_paras : dict | None
            Wasserstein2Loss 的参数字典。
        """
        super().__init__()
        self.height, self.width = image_shape
        self.time_steps = int(time_steps)
        self.smooth_sigma = float(smooth_sigma)
        self.reg_weight = float(reg_weight)
        self.w2 = Wasserstein2Loss(**(w2_paras or {}))
        self.sum_target = float(sum_target)
        self.device = device

        # 速度场参数：形状 (T, 2, H, W)
        self.velocity = nn.Parameter(
            torch.zeros(self.time_steps, 2, self.height, self.width, device=self.device)
        )

    def reset_velocity(self, scale: float = 0.0) -> None:
        """
        重置速度场参数。

        参数
        ----
        scale : float
            速度场初始化幅度（0 表示置零）。
        """
        with torch.no_grad():
            if scale == 0.0:
                self.velocity.zero_()
            else:
                self.velocity.normal_(mean=0.0, std=scale)

    def _integrate_flow(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        通过欧拉积分得到位移场。

        参数
        ----
        batch_size : int
            批大小 B。
        device : torch.device
            设备。
        dtype : torch.dtype
            数据类型。

        返回
        ----
        torch.Tensor
            位移场，形状 (B, 2, H, W)，单位为像素。
        """
        flow = torch.zeros(batch_size, 2, self.height, self.width, device=device, dtype=dtype)
        dt = 1.0 / max(self.time_steps, 1)
        for t in range(self.time_steps):
            v = self.velocity[t].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = smooth_field(v, self.smooth_sigma)
            v_at = sample_field(v, flow)
            flow = flow + dt * v_at
        return flow

    def _barycenter_loss(self, warped: Tensor, neighbors: Tensor, weights: Tensor) -> Tensor:
        """
        Wasserstein barycenter 目标：对相邻切片 W2 距离加权求和。

        参数
        ----
        warped : torch.Tensor
            形状 (B, 1, H, W)，当前切片经变形后的结果。
        neighbors : torch.Tensor
            相邻切片，可为 (N, H, W) / (N, 1, H, W) /
            (B, N, H, W) / (B, N, 1, H, W)。
        weights : torch.Tensor
            权重，可为 (N,) 或 (B, N)。

        返回
        ----
        torch.Tensor
            标量损失。
        """
        warped = ensure_4d(warped)
        if warped.shape[1] != 1:
            raise ValueError("warped 期望单通道 (B, 1, H, W)。")
        if not warped.is_cuda:
            raise ValueError("Wasserstein2Loss 仅支持 CUDA 张量，请将输入移至 GPU。")

        warped = normalize_distribution(warped, sum_target=self.sum_target)
        warped_2d = warped[:, 0, ...]  # (B, H, W)

        if neighbors.ndim == 3:
            batch_mode = False
        elif neighbors.ndim == 4:
            batch_mode = neighbors.shape[0] == warped.shape[0] and neighbors.shape[1] != 1
        elif neighbors.ndim == 5:
            batch_mode = True
        else:
            raise ValueError("neighbors 维度不合法。")

        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(warped.shape[0], -1)
        elif weights.ndim != 2:
            raise ValueError("weights 期望形状为 (N,) 或 (B, N)。")

        if not batch_mode:
            if neighbors.ndim == 3:
                neighbors_n = neighbors.unsqueeze(1)  # (N, 1, H, W)
            elif neighbors.ndim == 4 and neighbors.shape[1] == 1:
                neighbors_n = neighbors
            else:
                raise ValueError("非批处理模式下，neighbors 期望形状为 (N, H, W) 或 (N, 1, H, W)。")
            neighbors_n = normalize_distribution(neighbors_n, sum_target=self.sum_target)
            neighbors_2d = neighbors_n[:, 0, ...]  # (N, H, W)
            w = weights[0]
            if w.shape[0] != neighbors_2d.shape[0]:
                raise ValueError("weights 的长度与 neighbors 数量不一致。")
            w = w / (w.sum() + 1e-12)
            w2 = self.w2(warped_2d, neighbors_2d)  # (B, N)
            return (w2 * w.unsqueeze(0)).sum()

        loss = torch.zeros((), device=warped.device, dtype=warped.dtype)
        for b in range(warped.shape[0]):
            nb = neighbors[b]
            if nb.ndim == 3:
                nb = nb.unsqueeze(1)
            elif nb.ndim == 4 and nb.shape[1] != 1:
                nb = nb.unsqueeze(1)
            nb = normalize_distribution(nb, sum_target=self.sum_target)
            nb_2d = nb[:, 0, ...]
            w = weights[b]
            if w.shape[0] != nb_2d.shape[0]:
                raise ValueError("weights 的长度与 neighbors 数量不一致。")
            w = w / (w.sum() + 1e-12)
            w2 = self.w2(warped_2d[b : b + 1], nb_2d)  # (1, N)
            loss = loss + (w2.squeeze(0) * w).sum()
        return loss

    def forward(
        self,
        source: Tensor,
        neighbors: Tensor,
        weights: Tensor,
    ) -> Dict[str, Tensor]:
        """
        前向计算：得到变形结果与损失。

        参数
        ----
        source : torch.Tensor
            形状为 (H, W) / (1, H, W) / (B, 1, H, W)。
        neighbors : torch.Tensor
            相邻切片张量（见 _barycenter_loss 说明）。
        weights : torch.Tensor
            权重，形状为 (N,) 或 (B, N)。

        返回
        ----
        Dict[str, torch.Tensor]
            包含 warped, flow, loss_total, loss_w2, loss_reg。
        """
        source = ensure_4d(source)
        if source.shape[1] != 1:
            raise ValueError("source 期望单通道 (B, 1, H, W)。")
        if source.shape[-2:] != (self.height, self.width):
            raise ValueError("source 的空间尺寸与 LDDMM 初始化尺寸不一致。{} != {}".format(source.shape[-2:], (self.height, self.width)))

        flow = self._integrate_flow(source.shape[0], source.device, source.dtype)
        warped = warp_image(source, flow)

        loss_w2 = self._barycenter_loss(warped, neighbors, weights)
        loss_reg = gradient_l2(self.velocity) * self.reg_weight
        loss_total = loss_w2 + loss_reg

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            raise RuntimeError("损失出现 NaN/Inf，请检查输入数据或参数设置。")

        return {
            "warped": warped,
            "flow": flow,
            "loss_total": loss_total,
            "loss_w2": loss_w2,
            "loss_reg": loss_reg,
        }
