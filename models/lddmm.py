from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .utils import sample_field, smooth_field, warp_image, ensure_4d

Tensor = torch.Tensor


class LDDMM(nn.Module):
    """
    基于速度场的 2D LDDMM（仅负责生成变形后的切片与位移场）。
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        time_steps: int = 5,
        smooth_sigma: float = 1.0,
        device: torch.device = torch.device("cuda"),
        interp_mode: str = "bicubic",
        padding_mode: str = "border",
        align_corners: bool = True,
        mass_preserve: bool = True,
    ) -> None:
        """
        参数
        ----
        image_shape : Tuple[int, int]
            图像形状 (H, W)。
        time_steps : int
            时间离散步数 T。
        smooth_sigma : float
            速度场平滑标准差。
        device : torch.device
            设备。
        interp_mode : str
            重采样插值方式。
        padding_mode : str
            超出边界的采样策略。
        align_corners : bool
            grid_sample 的对齐方式。
        mass_preserve : bool
            是否进行强度守恒（乘以雅可比行列式）。
        """
        super().__init__()
        self.height, self.width = image_shape
        self.time_steps = int(time_steps)
        self.smooth_sigma = float(smooth_sigma)
        self.device = device
        self.interp_mode = interp_mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.mass_preserve = bool(mass_preserve)

        self.velocity = nn.Parameter(
            torch.zeros(self.time_steps, 2, self.height, self.width, device=self.device)
        )

    def reset_velocity(self, scale: float = 0.0) -> None:
        with torch.no_grad():
            if scale == 0.0:
                self.velocity.zero_()
            else:
                self.velocity.normal_(mean=0.0, std=scale)

    def _integrate_flow(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        flow = torch.zeros(batch_size, 2, self.height, self.width, device=device, dtype=dtype)
        dt = 1.0 / max(self.time_steps, 1)
        for t in range(self.time_steps):
            v = self.velocity[t].unsqueeze(0).expand(batch_size, -1, -1, -1)
            v = smooth_field(v, self.smooth_sigma)
            v_at = sample_field(v, flow, mode="bilinear")
            flow = flow + dt * v_at
        return flow

    def forward(self, source: Tensor) -> Dict[str, Tensor]:
        """
        参数
        ----
        source : torch.Tensor
            形状为 (H, W) / (1, H, W) / (B, 1, H, W)。

        返回
        ----
        Dict[str, torch.Tensor]
            包含 warped, flow。
        """
        source = ensure_4d(source)
        if source.shape[1] != 1:
            raise ValueError("source 期望单通道 (B, 1, H, W)。")
        if source.shape[-2:] != (self.height, self.width):
            raise ValueError("source 的空间尺寸与 LDDMM 初始化尺寸不一致。")

        flow = self._integrate_flow(source.shape[0], source.device, source.dtype)
        warped = warp_image(
            source,
            flow,
            mode=self.interp_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            mass_preserve=self.mass_preserve,
        )
        return {"warped": warped, "flow": flow}
