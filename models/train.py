from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from .data import SlicePreprocessor
from .lddmm import LDDMM
from .utils import gradient_l2
from .wasserstein2_loss import Wasserstein2Loss

Tensor = torch.Tensor


class EarlyStopping:
    """
    基于损失波动的早停策略。
    """

    def __init__(self, window: int = 10, min_delta: float = 1e-4) -> None:
        self.window = int(window)
        self.min_delta = float(min_delta)
        self.losses: List[float] = []

    def step(self, loss: float) -> bool:
        self.losses.append(loss)
        if len(self.losses) < self.window:
            return False
        recent = self.losses[-self.window :]
        return (max(recent) - min(recent)) < self.min_delta


def _build_output_dir(base_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _setup_logger(out_dir: str) -> logging.Logger:
    logger = logging.getLogger(f"ssem_train_{out_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(os.path.join(out_dir, "train.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def _weights_positive(weights: Sequence[float], r: int) -> List[float]:
    if len(weights) == r:
        return list(weights)
    if len(weights) == 2 * r:
        return list(weights[r:])
    raise ValueError("weights 长度应为 r 或 2r。")


def _temporal_smoothness(v: Tensor) -> Tensor:
    if v.shape[0] < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    return (v[1:] - v[:-1]).pow(2).mean()


def get_loss(
    name: str,
    source: Tensor,
    target: Tensor,
    w2: Optional[Wasserstein2Loss] = None,
) -> Tensor:
    """
    根据名称计算配准损失。

    参数
    ----
    name : str
        损失名称，支持 "w2" / "wasserstein" / "l2"。
    source : torch.Tensor
        形状为 (B, H, W)。
    target : torch.Tensor
        形状为 (N, H, W)。
    w2 : Wasserstein2Loss | None
        Wasserstein 损失实例。
    """
    name = name.lower()
    if name in ("w2", "W2", "wasserstein"):
        if w2 is None:
            raise ValueError("使用 Wasserstein 需要提供 w2 实例。")
        return w2(source, target)
    if name in ("l2", "L2", "mse"):
        if source.ndim != 3 or target.ndim != 3:
            raise ValueError("L2 损失期望 source/target 形状为 (B, H, W)/(N, H, W)。")
        diff = source.unsqueeze(1) - target.unsqueeze(0)
        return diff.pow(2).mean(dim=(-2, -1)).squeeze(0)
    raise ValueError(f"未知损失类型: {name}")


def train_global_alignment(
    slices: Tensor,
    output_dir: str,
    r: int = 3,
    weights: Optional[Sequence[float]] = None,
    loss_func: str = "w2",
    epochs: int = 200,
    lr: float = 1e-3,
    reg_grad: float = 1e-3,
    reg_time: float = 1e-3,
    reg_l2: float = 1e-4,
    early_stop_window: int = 10,
    early_stop_delta: float = 1e-4,
    save_interval: int = 10,
    device: torch.device = torch.device("cuda"),
    w2_paras: Optional[Dict[str, float]] = None,
    preprocess: Optional[SlicePreprocessor] = None,
    lddmm_kwargs: Optional[Dict[str, object]] = None,
) -> str:
    """
    全局切片对齐训练框架。

    参数
    ----
    slices : torch.Tensor
        形状为 (N, H, W) / (N, 1, H, W)。
    output_dir : str
        输出目录（函数内部会创建时间子目录）。
    r : int
        邻域窗口大小。
    weights : Sequence[float] | None
        长度为 2r 或 r 的权重序列。
    epochs : int
        训练轮数。
    lr : float
        学习率。
    reg_grad : float
        gradient_l2 正则权重。
    reg_time : float
        时间相邻速度场正则权重。
    reg_l2 : float
        速度场 L2 正则权重。
    early_stop_window : int
        早停窗口。
    early_stop_delta : float
        早停阈值。
    save_interval : int
        周期性保存间隔。
    device : torch.device
        训练设备。
    w2_paras : dict | None
        Wasserstein2Loss 参数。
    preprocess : SlicePreprocessor | None
        预处理器（若为 None 则使用默认）。
    lddmm_kwargs : dict | None
        LDDMM 额外参数。

    返回
    ----
    str
        训练输出目录。
    """
    out_dir = _build_output_dir(output_dir)
    logger = _setup_logger(out_dir)
    logger.info("输出目录: %s", out_dir)

    if weights is None:
        weights = [0.05, 0.15, 0.3, 0.3, 0.15, 0.05]
    weights_pos = _weights_positive(weights, r)

    if preprocess is None:
        preprocess = SlicePreprocessor()
    slices = preprocess.preprocess(slices).cpu()
    logger.info("preprocess slices shape: %s", slices.shape)
    
    n_slices, _, h, w = slices.shape

    lddmm_kwargs = lddmm_kwargs or {}
    models = [
        LDDMM(image_shape=(h, w), device=device, **lddmm_kwargs).to(device)
        for _ in range(n_slices)
    ]
    params = [p for m in models for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=lr)
    w2 = Wasserstein2Loss(**(w2_paras or {}))

    early_stop = EarlyStopping(window=early_stop_window, min_delta=early_stop_delta)
    best_loss = float("inf")
    best_path = os.path.join(out_dir, "best_models.pt")
    best_flow_path = os.path.join(out_dir, "best_flows.pt")

    try:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad(set_to_none=True)

            warped_list: List[Tensor] = []
            flow_list: List[Tensor] = []
            for k in range(n_slices):
                source = slices[k : k + 1].to(device, non_blocking=True)
                out = models[k](source)
                warped_list.append(out["warped"])
                flow_list.append(out["flow"])
            
            # logger.info("forward model completed.")

            loss_w2 = torch.zeros((), device=device)
            for k in range(n_slices):
                max_off = min(r, n_slices - 1 - k)
                if max_off <= 0:
                    continue
                w = torch.tensor(weights_pos[:max_off], device=device, dtype=warped_list[k].dtype)
                w = w / (w.sum() + 1e-12)
                neighbors = torch.cat(
                    [warped_list[k + off] for off in range(1, max_off + 1)], dim=0
                )
                loss_mat = get_loss(
                    loss_func,
                    warped_list[k][:, 0, ...],
                    neighbors[:, 0, ...],
                    w2=w2,
                )
                loss_w2 = loss_w2 + (loss_mat.squeeze(0) * w).sum()
            
            
            # logger.info("W2 input shapes: ", warped_list[k][:, 0, ...].shape, neighbors[:, 0, ...].shape)

            loss_reg = torch.zeros((), device=device)
            for m in models:
                v = m.velocity
                loss_reg = loss_reg + reg_grad * gradient_l2(v)
                loss_reg = loss_reg + reg_time * _temporal_smoothness(v)
                loss_reg = loss_reg + reg_l2 * v.pow(2).mean()

            loss_total = loss_w2 + loss_reg
            if torch.isnan(loss_total) or torch.isinf(loss_total):
                raise RuntimeError("损失出现 NaN/Inf，请检查输入数据或参数。")

            loss_total.backward()
            optimizer.step()

            loss_total_val = float(loss_total.detach().cpu())
            loss_w2_val = float(loss_w2.detach().cpu())
            loss_reg_val = float(loss_reg.detach().cpu())
            logger.info(
                "epoch=%d loss_total=%.6f loss_w2=%.6f loss_reg=%.6f",
                epoch,
                loss_total_val,
                loss_w2_val,
                loss_reg_val,
            )

            if loss_total_val < best_loss:
                best_loss = loss_total_val
                torch.save(
                    {
                        "epoch": epoch,
                        "loss": best_loss,
                        "models": [m.state_dict() for m in models],
                    },
                    best_path,
                )
                torch.save([f.detach().cpu() for f in flow_list], best_flow_path)

            if save_interval > 0 and epoch % save_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "loss": loss_total_val,
                        "models": [m.state_dict() for m in models],
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(out_dir, f"checkpoint_{epoch}.pt"),
                )

            if early_stop.step(loss_total_val):
                logger.info("早停触发: 波动小于阈值，停止训练。")
                break

    except Exception:
        logger.exception("训练异常终止。")
        raise

    return out_dir
