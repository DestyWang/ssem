from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .utils import ensure_4d, normalize_distribution

Tensor = torch.Tensor


class SlicePreprocessor:
    """
    体电镜切片预处理器：支持重采样、归一化与标准化等常用操作。
    """

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        sum_target: float = 1e7,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        interp_mode: str = "bilinear",
        align_corners: bool = False,
    ) -> None:
        """
        参数
        ----
        target_size : tuple | None
            目标尺寸 (H, W)，与 scale_factor 二选一。
        scale_factor : float | None
            缩放倍数，与 target_size 二选一。
        sum_target : float
            归一化的空间总和。
        mean : float | None
            指定均值，若为 None 则不做均值调整。
        std : float | None
            指定标准差，若为 None 则不做标准化。
        clamp_min : float | None
            下限裁剪。
        clamp_max : float | None
            上限裁剪。
        interp_mode : str
            重采样插值方式。
        align_corners : bool
            插值对齐方式。
        """
        self.target_size = target_size
        self.scale_factor = scale_factor
        self.sum_target = float(sum_target)
        self.mean = mean
        self.std = std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.interp_mode = interp_mode
        self.align_corners = align_corners

    def _to_tensor(self, x: Union[Tensor, "numpy.ndarray"]) -> Tensor:
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def _resize(self, x: Tensor) -> Tensor:
        if self.target_size is None and self.scale_factor is None:
            return x
        return F.interpolate(
            x,
            size=self.target_size,
            scale_factor=self.scale_factor,
            mode=self.interp_mode,
            align_corners=self.align_corners if self.interp_mode != "nearest" else None,
        )

    def _clamp(self, x: Tensor) -> Tensor:
        if self.clamp_min is None and self.clamp_max is None:
            return x
        return torch.clamp(x, min=self.clamp_min, max=self.clamp_max)

    def _standardize(self, x: Tensor) -> Tensor:
        if self.mean is None and self.std is None:
            return x
        mean = self.mean if self.mean is not None else x.mean()
        std = self.std if self.std is not None else x.std().clamp_min(1e-12)
        return (x - mean) / std

    def _normalize_sum(self, x: Tensor) -> Tensor:
        return normalize_distribution(x, sum_target=self.sum_target)

    def preprocess(self, x: Union[Tensor, "numpy.ndarray"]) -> Tensor:
        """
        预处理入口。

        参数
        ----
        x : Tensor | np.ndarray
            形状为 (H,W) / (N,H,W) / (N,1,H,W)。

        返回
        ----
        Tensor
            形状为 (N,1,H,W)。
        """
        x = self._to_tensor(x).float()
        x = ensure_4d(x)
        x = self._resize(x)
        x = self._clamp(x)
        x = self._standardize(x)
        x = self._normalize_sum(x)
        return x

    def iter_batches(self, x: Tensor, batch_size: int) -> Iterator[Tensor]:
        """
        以 batch 方式迭代输出，降低显存占用。

        参数
        ----
        x : torch.Tensor
            形状为 (N,1,H,W)。
        batch_size : int
            批大小。
        """
        if x.ndim != 4:
            raise ValueError("iter_batches 期望输入形状为 (N,1,H,W)。")
        n = x.shape[0]
        for i in range(0, n, batch_size):
            yield x[i : i + batch_size]
