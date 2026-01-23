# SSEM

### Theory
#### 方法1 （旧版方法，已废弃）
对每张切片$S_k$单独配准，邻居切片均使用原始图片：
$$
\Phi_k^* = \arg\min_{\Phi_k} \sum_{i=-r, i\neq 0}^{r}\lambda_i W_{2,\epsilon}^2(\Phi_k(S_k), S_{k+i}) +\text{Reg}(\Phi_k)
$$
其中$r$是窗口大小，$\lambda_i$是权重，$\text{Reg}(\cdot)$是对畸变场的正则化项，确保畸变场符合一些物理约束。

#### 方法2 （目前默认的方法）
直接全局配准，使用的是形变场作用后的邻居切片，互相之间有关联。
$$
\{\Phi_k^*\} = \arg\min_{\{\Phi_k\}} \sum_{i=-r, i\neq 0}^{r}\lambda_i W_{2,\epsilon}^2(\Phi_k(S_k), \Phi_{k+i}(S_{k+i})) +\text{Reg}(\{\Phi_k\})
$$

由于对称性，问题可以消除一半的冗余计算，得到
$$
\{\Phi_k^*\} = \arg\min_{\{\Phi_k\}} \sum_{i=1}^{r}\lambda_i W_{2,\epsilon}^2(\Phi_k(S_k), \Phi_{k+i}(S_{k+i})) +\text{Reg}(\{\Phi_k\})
$$

注意当邻居不存在时（下标不在$[1,N]$范围内），则不考虑当前切片，对求和的贡献为零。

### LDDMM 说明（`deform.py`）

**目标**  
对每张切片估计畸变场，使变形后的切片与相邻切片的加权 Wasserstein 距离最小：

\[
\Phi_k^* = \arg\min_{\Phi_k} \sum_{i\neq 0}\lambda_i W_{2,\epsilon}^2(\Phi_k(S_k), S_{k+i}) + \text{Reg}(\Phi_k)
\]

**实现要点**
- 使用时间离散的速度场 `v_t`（形状 `(T, 2, H, W)`）通过欧拉积分得到位移场。
- 配准损失采用 Wasserstein barycenter 目标：对相邻切片的 W2 加权求和。
- 正则项采用速度场梯度的 L2 惩罚。

**主要接口**
- `LDDMM(image_shape, time_steps, smooth_sigma, reg_weight, w2_paras)`
  - `image_shape: Tuple[int, int]`：`(H, W)`。
  - `time_steps: int`：时间离散步数 `T`。
  - `smooth_sigma: float`：速度场平滑的高斯核标准差。
  - `reg_weight: float`：正则化权重。
  - `w2_paras: dict`：`Wasserstein2Loss` 参数（如 `eps`, `maxiter` 等）。
- `forward(source, neighbors, weights) -> Dict[str, Tensor]`
  - `source: Tensor` 形状 `(H, W)` / `(1, H, W)` / `(B, 1, H, W)`。
  - `neighbors: Tensor` 形状 `(N, H, W)` / `(N, 1, H, W)` / `(B, N, H, W)` / `(B, N, 1, H, W)`。
  - `weights: Tensor` 形状 `(N,)` 或 `(B, N)`。
  - 返回字典包含：`warped`, `flow`, `loss_total`, `loss_w2`, `loss_reg`。

**注意事项**
- `Wasserstein2Loss` 仅支持 CUDA 张量，训练/推理需使用 GPU。
- 目标默认假设切片为非负分布，内部会执行归一化。
