from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def flow_to_normalized(flow: Tensor, height: int, width: int) -> Tensor:
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


def jacobian_determinant(flow: Tensor, eps: float = 1e-6) -> Tensor:
    """
    计算位移场对应的雅可比行列式 det(Dphi)。

    参数
    ----
    flow : torch.Tensor
        形状为 (B, 2, H, W)，单位为像素。
    eps : float
        稳定项，避免极小或负值。

    返回
    ----
    torch.Tensor
        形状为 (B, 1, H, W)。
    """
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("jacobian_determinant 期望 flow 形状为 (B, 2, H, W)。")

    u = flow[:, 0]
    v = flow[:, 1]

    du_dx = u[:, :, 1:] - u[:, :, :-1]
    du_dx = F.pad(du_dx, (0, 1, 0, 0), mode="replicate")
    du_dy = u[:, 1:, :] - u[:, :-1, :]
    du_dy = F.pad(du_dy, (0, 0, 0, 1), mode="replicate")

    dv_dx = v[:, :, 1:] - v[:, :, :-1]
    dv_dx = F.pad(dv_dx, (0, 1, 0, 0), mode="replicate")
    dv_dy = v[:, 1:, :] - v[:, :-1, :]
    dv_dy = F.pad(dv_dy, (0, 0, 0, 1), mode="replicate")

    det = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
    det = torch.clamp(det, min=eps)
    return det.unsqueeze(1)


def warp_image(
    image: Tensor,
    flow: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
    mass_preserve: bool = False,
) -> Tensor:
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
    flow_norm = flow_to_normalized(flow, h, w)
    sample_grid = grid + flow_norm
    warped = F.grid_sample(
        image,
        sample_grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if mass_preserve:
        jac = jacobian_determinant(flow)
        warped = warped * jac
    return warped


def sample_field(field: Tensor, flow: Tensor, mode: str = "bilinear") -> Tensor:
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
    return warp_image(field, flow, mode=mode, padding_mode="border", align_corners=True, mass_preserve=False)


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


def plot_2d_velocity_field(
    velocity: Tensor,
    t: int = 0,
    stride: int = 8,
    scale: float = 1.0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """
    可视化二维速度场（quiver）。

    参数
    ----
    velocity : torch.Tensor
        形状为 (T, 2, H, W)。
    t : int
        时间点索引。
    stride : int
        下采样步长，控制箭头密度。
    scale : float
        箭头缩放系数。
    ax : matplotlib.axes.Axes | None
        外部传入坐标轴。
    title : str | None
        图标题。
    figsize : tuple | None
        图像大小。

    返回
    ----
    matplotlib.axes.Axes
    """
    if velocity.ndim != 4 or velocity.shape[1] != 2:
        raise ValueError("velocity 期望形状为 (T, 2, H, W)。")
    if not (0 <= t < velocity.shape[0]):
        raise ValueError("t 超出范围。")
    v = velocity[t].detach().cpu().float()
    u = v[0]
    w = v[1]
    h, wdim = u.shape
    ys = torch.arange(0, h, stride)
    xs = torch.arange(0, wdim, stride)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.quiver(
        grid_x.numpy(),
        grid_y.numpy(),
        u[grid_y, grid_x].numpy(),
        w[grid_y, grid_x].numpy(),
        angles="xy",
        scale_units="xy",
        scale=1.0 / max(scale, 1e-8),
        width=0.0025,
    )
    ax.invert_yaxis()
    if title is None:
        title = f"Velocity field at t={t}"
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax


def plot_flow_grid(
    flow: Tensor,
    grid_spacing: int = 16,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Axes:
    """
    可视化形变场对规则网格的扭曲效果。

    参数
    ----
    flow : torch.Tensor
        形状为 (2, H, W) 或 (B, 2, H, W)。
    grid_spacing : int
        网格线间距（像素）。
    ax : matplotlib.axes.Axes | None
        外部传入坐标轴。
    title : str | None
        图标题。
    figsize : tuple | None
        图像大小。
    返回
    ----
    matplotlib.axes.Axes
    """
    if flow.ndim == 4:
        flow = flow[0]
    if flow.ndim != 3 or flow.shape[0] != 2:
        raise ValueError("flow 期望形状为 (2, H, W) 或 (B, 2, H, W)。")
    u = flow[0].detach().cpu().float()
    v = flow[1].detach().cpu().float()
    h, w = u.shape

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ys = torch.arange(0, h, grid_spacing)
    xs = torch.arange(0, w, grid_spacing)

    for y in ys:
        x_coords = torch.arange(0, w)
        y_coords = torch.full((w,), float(y))
        x_warp = x_coords + u[y.long(), x_coords]
        y_warp = y_coords + v[y.long(), x_coords]
        ax.plot(x_warp.numpy(), y_warp.numpy(), color="C0", linewidth=0.8)

    for x in xs:
        y_coords = torch.arange(0, h)
        x_coords = torch.full((h,), float(x))
        x_warp = x_coords + u[y_coords, x.long()]
        y_warp = y_coords + v[y_coords, x.long()]
        ax.plot(x_warp.numpy(), y_warp.numpy(), color="C0", linewidth=0.8)

    ax.invert_yaxis()
    if title is None:
        title = "Warped grid"
    ax.set_title(title)
    ax.set_aspect("equal")
    return ax
