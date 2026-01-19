from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def ensure_4d(x: Tensor) -> Tensor:
    """
    将输入转为 (B, C, H, W) 形状的张量。

    参数
    ----
    x : torch.Tensor
        形状为 (H, W) 或 (C, H, W) 或 (B, C, H, W)。

    返回
    ----
    torch.Tensor
        形状为 (B, C, H, W)。
    """
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"期望 2~4 维张量，但得到 {x.ndim} 维。")


def normalize_distribution(x: Tensor, eps: float = 1e-12, sum_target: float = 1e7) -> Tensor:
    """
    将张量裁剪为非负并按空间维度归一化为指定和的分布。

    参数
    ----
    x : torch.Tensor
        形状为 (B, C, H, W)。
    eps : float
        数值稳定常数。
    sum_target : float
        归一化后的空间维度总和（默认 1e7）。

    返回
    ----
    torch.Tensor
        形状为 (B, C, H, W)。
    """
    if x.ndim != 4:
        raise ValueError("normalize_distribution 期望输入形状为 (B, C, H, W)。")
    x = torch.clamp(x, min=0.0)
    denom = x.sum(dim=(-2, -1), keepdim=True) + eps
    scale = sum_target / denom
    return x * scale


def identity_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    生成 grid_sample 需要的标准化坐标网格。

    参数
    ----
    height : int
        图像高度 H。
    width : int
        图像宽度 W。
    device : torch.device
        张量所在设备。
    dtype : torch.dtype
        张量数据类型。

    返回
    ----
    torch.Tensor
        形状为 (1, H, W, 2)，范围 [-1, 1]。
    """
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0)


def _flow_to_normalized(flow: Tensor, height: int, width: int) -> Tensor:
    """
    将像素位移场转换到 [-1, 1] 标准化坐标。

    参数
    ----
    flow : torch.Tensor
        形状为 (B, 2, H, W)，单位为像素。
    height : int
        H。
    width : int
        W。

    返回
    ----
    torch.Tensor
        形状为 (B, H, W, 2)。
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow 期望形状为 (B, 2, H, W)。")
    norm_x = 2.0 / max(width - 1, 1)
    norm_y = 2.0 / max(height - 1, 1)
    flow_x = flow[:, 0] * norm_x
    flow_y = flow[:, 1] * norm_y
    return torch.stack([flow_x, flow_y], dim=-1)


def warp_image(image: Tensor, flow: Tensor) -> Tensor:
    """
    使用位移场对图像进行变形。

    参数
    ----
    image : torch.Tensor
        形状为 (B, C, H, W)。
    flow : torch.Tensor
        形状为 (B, 2, H, W)，单位为像素。

    返回
    ----
    torch.Tensor
        形状为 (B, C, H, W)。
    """
    if image.ndim != 4:
        raise ValueError("warp_image 期望 image 形状为 (B, C, H, W)。")
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("warp_image 期望 flow 形状为 (B, 2, H, W)。")
    b, _, h, w = image.shape
    grid = identity_grid(h, w, image.device, image.dtype)
    flow_norm = _flow_to_normalized(flow, h, w)
    sample_grid = grid + flow_norm
    return F.grid_sample(
        image,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def sample_field(field: Tensor, flow: Tensor) -> Tensor:
    """
    在位移后的坐标处采样向量场。

    参数
    ----
    field : torch.Tensor
        形状为 (B, C, H, W)。
    flow : torch.Tensor
        形状为 (B, 2, H, W)，单位为像素。

    返回
    ----
    torch.Tensor
        形状为 (B, C, H, W)。
    """
    return warp_image(field, flow)


def gaussian_kernel2d(sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    生成二维高斯核。

    参数
    ----
    sigma : float
        标准差。
    device : torch.device
        设备。
    dtype : torch.dtype
        数据类型。

    返回
    ----
    torch.Tensor
        形状为 (1, 1, K, K)。
    """
    if sigma <= 0:
        raise ValueError("sigma 必须为正数。")
    radius = int(3 * sigma)
    size = 2 * radius + 1
    coords = torch.arange(size, device=device, dtype=dtype) - radius
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, size, size)


def smooth_field(field: Tensor, sigma: float) -> Tensor:
    """
    使用高斯核对向量场进行平滑。

    参数
    ----
    field : torch.Tensor
        形状为 (B, C, H, W)。
    sigma : float
        高斯核标准差。

    返回
    ----
    torch.Tensor
        形状为 (B, C, H, W)。
    """
    if sigma <= 0:
        return field
    if field.ndim != 4:
        raise ValueError("smooth_field 期望 field 形状为 (B, C, H, W)。")
    kernel = gaussian_kernel2d(sigma, field.device, field.dtype)
    kernel = kernel.repeat(field.shape[1], 1, 1, 1)
    padding = kernel.shape[-1] // 2
    return F.conv2d(field, kernel, padding=padding, groups=field.shape[1])


def gradient_l2(field: Tensor) -> Tensor:
    """
    计算向量场的梯度正则项 ||∇v||^2。

    参数
    ----
    field : torch.Tensor
        形状为 (B, C, H, W)。

    返回
    ----
    torch.Tensor
        标量张量。
    """
    if field.ndim != 4:
        raise ValueError("gradient_l2 期望 field 形状为 (B, C, H, W)。")
    dx = field[..., 1:] - field[..., :-1]
    dy = field[..., 1:, :] - field[..., :-1, :]
    return (dx.pow(2).mean() + dy.pow(2).mean())
