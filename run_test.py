import argparse
import os
import sys
from typing import List, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("/home/bcl/wanghongyu/wanghongyu_humx/dynamic/ssem/models")
sys.path.append("/home/bcl/wanghongyu/wanghongyu_humx/dynamic/tools")

from models.data import SlicePreprocessor
from models.train import train_global_alignment
from models.utils import warp_image


def _parse_weights(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _load_hdf5_volume(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        raw_data = f["volumes/raw"]
        return np.array(raw_data)


def _downsample(volume: np.ndarray, scale_h: int, scale_w: int) -> np.ndarray:
    x = torch.from_numpy(volume).unsqueeze(1).float()
    pool = torch.nn.AvgPool2d(kernel_size=(scale_h, scale_w))
    return pool(x).squeeze(1).numpy()


def _select_indices(length: int, k: int) -> List[int]:
    if k <= 1:
        return [length // 2]
    idx = torch.linspace(0, length - 1, steps=k).round().long().tolist()
    return [int(i) for i in idx]


def _plot_axis_compare(
    orig: torch.Tensor,
    warped: torch.Tensor,
    axis: int,
    k: int,
    save_path: str,
) -> None:
    idx = _select_indices(orig.shape[axis], k)
    fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6))
    if len(idx) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, i in enumerate(idx):
        if axis == 0:
            o = orig[i]
            w = warped[i]
        elif axis == 1:
            o = orig[:, i, :]
            w = warped[:, i, :]
        else:
            o = orig[:, :, i]
            w = warped[:, :, i]
        axes[0, j].imshow(o, cmap="gray")
        axes[0, j].set_title(f"orig a{axis}={i}")
        axes[0, j].axis("off")
        axes[1, j].imshow(w, cmap="gray")
        axes[1, j].set_title(f"warp a{axis}={i}")
        axes[1, j].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--down_sample", action="store_true")
    parser.add_argument("--scale_h", type=int, default=4)
    parser.add_argument("--scale_w", type=int, default=4)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument(
        "--weights",
        type=str,
        default="0.05,0.15,0.3,0.3,0.15,0.05",
        help="逗号分隔，长度为 2r 或 r",
    )
    parser.add_argument("--loss_func", type=str, default="w2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--reg_grad", type=float, default=1e-2)
    parser.add_argument("--reg_time", type=float, default=1e-2)
    parser.add_argument("--reg_l2", type=float, default=1e-2)
    parser.add_argument("--early_stop_window", type=int, default=10)
    parser.add_argument("--early_stop_delta", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--w2_eps", type=float, default=1e-8)
    parser.add_argument("--w2_maxiter", type=int, default=100)
    parser.add_argument("--time_steps", type=int, default=10)
    parser.add_argument("--smooth_sigma", type=float, default=1.0)
    parser.add_argument("--interp_mode", type=str, default="bicubic")
    parser.add_argument("--padding_mode", type=str, default="border")
    parser.add_argument("--align_corners", action="store_true")
    parser.add_argument("--mass_preserve", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="每个维度显示的切片数")
    args = parser.parse_args()

    device = torch.device(args.device)
    weights = _parse_weights(args.weights)

    dataset = _load_hdf5_volume(args.data_path)
    if args.down_sample:
        dataset = _downsample(dataset, args.scale_h, args.scale_w)
    data_t = torch.from_numpy(dataset).float() / 255.0

    sum_target = data_t.sum(dim=(1, 2)).mean().item()
    slicepreprocessor = SlicePreprocessor(
        target_size=None,
        scale_factor=None,
        sum_target=sum_target,
        mean=None,
        std=None,
        clamp_min=None,
        clamp_max=None,
        interp_mode="bilinear",
        align_corners=False,
    )

    out_dir = train_global_alignment(
        slices=data_t.unsqueeze(1),
        output_dir=args.output_dir,
        r=args.r,
        weights=weights,
        loss_func=args.loss_func,
        epochs=args.epochs,
        lr=args.lr,
        reg_grad=args.reg_grad,
        reg_time=args.reg_time,
        reg_l2=args.reg_l2,
        early_stop_window=args.early_stop_window,
        early_stop_delta=args.early_stop_delta,
        save_interval=args.save_interval,
        device=device,
        w2_paras={"eps": args.w2_eps, "maxiter": args.w2_maxiter},
        preprocess=slicepreprocessor,
        lddmm_kwargs={
            "time_steps": args.time_steps,
            "smooth_sigma": args.smooth_sigma,
            "interp_mode": args.interp_mode,
            "padding_mode": args.padding_mode,
            "align_corners": args.align_corners,
            "mass_preserve": args.mass_preserve,
        },
    )

    flows_path = os.path.join(out_dir, "best_flows.pt")
    flows = torch.load(flows_path, map_location=device)
    preprocessed = slicepreprocessor.preprocess(data_t.unsqueeze(1)).to(device)

    warped_list = []
    for k, flow in enumerate(flows):
        src = preprocessed[k : k + 1]
        flow = flow.to(device)
        warped = warp_image(
            src,
            flow,
            mode=args.interp_mode,
            padding_mode=args.padding_mode,
            align_corners=args.align_corners,
            mass_preserve=args.mass_preserve,
        )
        warped_list.append(warped)
    warped = torch.cat(warped_list, dim=0).squeeze(1).detach().cpu()
    orig = preprocessed.squeeze(1).detach().cpu()

    os.makedirs(out_dir, exist_ok=True)
    _plot_axis_compare(orig, warped, axis=0, k=args.k, save_path=os.path.join(out_dir, "compare_axis0.png"))
    _plot_axis_compare(orig, warped, axis=1, k=args.k, save_path=os.path.join(out_dir, "compare_axis1.png"))
    _plot_axis_compare(orig, warped, axis=2, k=args.k, save_path=os.path.join(out_dir, "compare_axis2.png"))


if __name__ == "__main__":
    main()