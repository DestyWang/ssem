"""Evaluation metrics for serial section alignment.

All functions support:
1) single stack input with shape (N, H, W): compute metric on adjacent pairs;
2) two stacks with same shape (N, H, W): compute metric on corresponding pairs.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _to_numpy(data: ArrayLike, name: str) -> np.ndarray:
    """Convert torch/numpy input to numpy array with dtype float64."""
    if torch is not None and isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W), but got {arr.shape}.")
    return arr.astype(np.float64, copy=False)


def _prepare_pairs(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare pair arrays for pair-wise metric computation.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            left/right pair stacks, both with shape (M, H, W),
            M = N - 1 (adjacent mode) or M = N (reference mode).
    """
    a = _to_numpy(stack, "stack")
    if reference_stack is None:
        if a.shape[0] < 2:
            raise ValueError("stack must contain at least 2 slices in adjacent mode.")
        return a[:-1], a[1:]

    b = _to_numpy(reference_stack, "reference_stack")
    if a.shape != b.shape:
        raise ValueError(
            f"stack and reference_stack must have same shape, got {a.shape} and {b.shape}."
        )
    return a, b


def _safe_corrcoef(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Pearson correlation robustly for flattened arrays."""
    x0 = x.reshape(-1) - x.mean()
    y0 = y.reshape(-1) - y.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)) + eps
    return float(np.sum(x0 * y0) / denom)


def compute_ncc(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> float:
    """Compute mean NCC score for a stack.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).
        eps (float): numerical stability term, shape ().

    Returns:
        float: mean NCC value over all evaluated pairs, shape ().
    """
    a, b = _prepare_pairs(stack, reference_stack)
    ncc_values = [_safe_corrcoef(ai, bi, eps=eps) for ai, bi in zip(a, b)]
    return float(np.mean(ncc_values))


def compute_ssim(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    """Compute mean global SSIM score for a stack.

    Notes:
        This is a global SSIM (single-window) variant, lightweight for quick monitoring.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).
        k1 (float): SSIM constant coefficient, shape ().
        k2 (float): SSIM constant coefficient, shape ().
        eps (float): numerical stability term, shape ().

    Returns:
        float: mean SSIM value over all evaluated pairs, shape ().
    """
    a, b = _prepare_pairs(stack, reference_stack)
    ssim_values = []
    for ai, bi in zip(a, b):
        data_min = min(float(ai.min()), float(bi.min()))
        data_max = max(float(ai.max()), float(bi.max()))
        data_range = max(data_max - data_min, eps)

        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2

        mu_a = ai.mean()
        mu_b = bi.mean()
        sigma_a2 = np.mean((ai - mu_a) ** 2)
        sigma_b2 = np.mean((bi - mu_b) ** 2)
        sigma_ab = np.mean((ai - mu_a) * (bi - mu_b))

        numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
        denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a2 + sigma_b2 + c2) + eps
        ssim_values.append(float(numerator / denominator))
    return float(np.mean(ssim_values))


def compute_mi(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike] = None,
    bins: int = 64,
    eps: float = 1e-12,
) -> float:
    """Compute mean Mutual Information (MI) score for a stack.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).
        bins (int): number of histogram bins for MI estimation, shape ().
        eps (float): numerical stability term, shape ().

    Returns:
        float: mean MI value over all evaluated pairs, shape ().
    """
    a, b = _prepare_pairs(stack, reference_stack)
    mi_values = []
    for ai, bi in zip(a, b):
        hist_2d, _, _ = np.histogram2d(ai.reshape(-1), bi.reshape(-1), bins=bins)
        pxy = hist_2d / np.maximum(hist_2d.sum(), eps)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        px_py = px @ py
        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log((pxy[mask] + eps) / (px_py[mask] + eps)))
        mi_values.append(float(mi))
    return float(np.mean(mi_values))


def compute_gradient_correlation(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> float:
    """Compute mean Gradient Correlation (GC) score for a stack.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).
        eps (float): numerical stability term, shape ().

    Returns:
        float: mean GC value over all evaluated pairs, shape ().
    """
    a, b = _prepare_pairs(stack, reference_stack)
    gc_values = []
    for ai, bi in zip(a, b):
        grad_ay, grad_ax = np.gradient(ai)
        grad_by, grad_bx = np.gradient(bi)
        corr_x = _safe_corrcoef(grad_ax, grad_bx, eps=eps)
        corr_y = _safe_corrcoef(grad_ay, grad_by, eps=eps)
        gc_values.append(float(0.5 * (corr_x + corr_y)))
    return float(np.mean(gc_values))


def compute_phase_correlation_peak(
    stack: ArrayLike,
    reference_stack: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> float:
    """Compute mean phase-correlation peak value for a stack.

    Args:
        stack (ArrayLike): input stack, shape (N, H, W).
        reference_stack (Optional[ArrayLike]): optional reference stack, shape (N, H, W).
        eps (float): numerical stability term, shape ().

    Returns:
        float: mean phase-correlation peak over all evaluated pairs, shape ().
    """
    a, b = _prepare_pairs(stack, reference_stack)
    peak_values = []
    for ai, bi in zip(a, b):
        fa = np.fft.fft2(ai)
        fb = np.fft.fft2(bi)
        cross_power = fa * np.conj(fb)
        cross_power /= np.maximum(np.abs(cross_power), eps)
        corr = np.fft.ifft2(cross_power)
        peak_values.append(float(np.abs(corr).max()))
    return float(np.mean(peak_values))


__all__ = [
    "compute_ncc",
    "compute_ssim",
    "compute_mi",
    "compute_gradient_correlation",
    "compute_phase_correlation_peak",
]
