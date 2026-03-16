# Copyright (c) FlashOCC1.5. All rights reserved.
#
# INROCCHead: Implicit Neural Representation for Continuous 3D Occupancy
#
# 创新点：
#   将 FlashOCC 的 BEV 特征（B,C,Dy,Dx）输入给一个坐标网络（MLP+傅里叶位置编码），
#   对任意三维坐标 (x,y,z) 做连续化占据预测，打破固定体素分辨率的限制。
#   训练时随机采样可见体素坐标，推理时密集枚举全体素网格（可按 chunk 执行）。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.runner import BaseModule
from mmdet3d.models.builder import HEADS, build_loss
from ..losses.semkitti_loss import sem_scal_loss, geo_scal_loss
from ..losses.lovasz_softmax import lovasz_softmax, lovasz_softmax_flat


nusc_class_frequencies = np.array([
    944004,
    1897170,
    152386,
    2391677,
    16957802,
    724139,
    189027,
    2074468,
    413451,
    2384460,
    5916653,
    175883646,
    4275424,
    51393615,
    61411620,
    105975596,
    116424404,
    1892500630
])


class FourierPosEnc(nn.Module):
    """Fourier positional encoding for a 1-D scalar (e.g. z-coordinate).

    Encodes a value in range [0, 1] using sin/cos at log-spaced frequencies.
    Output dim = include_input + 2 * num_freqs (per input dimension).
    Here the input is 1-D (just z), so out_dim = 1 + 2*num_freqs.
    """

    def __init__(self, num_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        # log-spaced frequencies [1, 2, 4, ..., 2^(num_freqs-1)]
        self.register_buffer(
            'freqs',
            2.0 ** torch.arange(num_freqs).float() * math.pi
        )

    @property
    def out_dim(self) -> int:
        d = 1  # 1-D input
        base = 2 * d * self.num_freqs
        return base + d if self.include_input else base

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (..., 1)  normalized to [0, 1]
        Returns:
            (..., out_dim)
        """
        # (..., 1, num_freqs)
        phases = z.unsqueeze(-1) * self.freqs
        # (..., 2*num_freqs)
        enc = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        enc = enc.flatten(-2)
        if self.include_input:
            enc = torch.cat([z, enc], dim=-1)
        return enc


@HEADS.register_module()
class INROCCHead(BaseModule):
    """Implicit Neural Representation Occupancy Head.

    Instead of running a fixed Channel-to-Height 2D→3D mapping, this head
    queries the BEV feature map at continuous (x,y) positions via bilinear
    interpolation, then feeds a Fourier-encoded z coordinate together with
    the sampled feature into a small MLP to predict per-point class logits.

    Training: randomly samples `max_train_pts` voxel coordinates per sample
              from the camera-visible mask, enabling large virtual batches.
    Inference: enumerates the full Dx×Dy×Dz voxel grid (chunked for memory).

    Args:
        in_dim (int): channel dim of input BEV feature.
        hidden_dim (int): hidden dim of the coordinate MLP.
        num_freqs (int): number of Fourier frequencies for z encoding.
        num_classes (int): number of semantic classes.
        Dx, Dy, Dz (int): voxel grid dimensions.
        x_range, y_range, z_range (tuple): real-world coordinate ranges.
        max_train_pts (int|None): max valid voxels sampled per sample during
            training. None = use all valid voxels (may OOM for large batches).
        infer_chunk (int): number of voxels processed per forward chunk at
            inference to bound GPU memory.
        use_mask (bool): apply camera mask when computing loss.
        class_balance (bool): inverse-log-frequency class weighting.
        use_aux_losses (bool): add lovasz + scal losses (like BEVOCCHead2D_V2).
        loss_occ (dict): config for the occupancy cross-entropy loss.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        num_freqs: int = 8,
        num_classes: int = 18,
        Dx: int = 200,
        Dy: int = 200,
        Dz: int = 16,
        x_range: tuple = (-40.0, 40.0),
        y_range: tuple = (-40.0, 40.0),
        z_range: tuple = (-1.0, 5.4),
        max_train_pts: int = 8192,
        infer_chunk: int = 40000,
        use_mask: bool = True,
        class_balance: bool = True,
        use_aux_losses: bool = True,
        loss_occ: dict = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.Dx = Dx
        self.Dy = Dy
        self.Dz = Dz
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_classes = num_classes
        self.use_mask = use_mask
        self.max_train_pts = max_train_pts
        self.infer_chunk = infer_chunk
        self.use_aux_losses = use_aux_losses

        # ---- Fourier positional encoding for z ----
        self.z_pe = FourierPosEnc(num_freqs=num_freqs, include_input=True)
        z_feat_dim = self.z_pe.out_dim  # 1 + 2*num_freqs = 17

        # ---- Coordinate MLP ----
        # in_dim: BEV sampled feature (C) + z Fourier encoding
        mlp_in = in_dim + z_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ---- Class balance ----
        self.class_balance = class_balance
        if class_balance:
            cw = torch.from_numpy(
                1.0 / np.log(nusc_class_frequencies[:num_classes] + 0.001)
            ).float()
            self.register_buffer('cls_weights', cw)
        else:
            self.cls_weights = None

        # ---- Loss ----
        if loss_occ is None:
            loss_occ = dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
            )
        if class_balance and 'class_weight' not in loss_occ:
            loss_occ = dict(loss_occ)  # copy
            loss_occ['class_weight'] = self.cls_weights
        self.loss_occ_fn = build_loss(loss_occ)

        # ---- Pre-compute voxel grid coords for inference ----
        self._build_infer_grid()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_infer_grid(self):
        """Pre-compute all (N=Dx*Dy*Dz) coordinate tensors for inference."""
        ix = torch.arange(self.Dx, dtype=torch.float32)
        iy = torch.arange(self.Dy, dtype=torch.float32)
        iz = torch.arange(self.Dz, dtype=torch.float32)

        # grid_sample x/y in [-1, 1]: x → Dx dim (width), y → Dy dim (height)
        gx = (ix + 0.5) / self.Dx * 2.0 - 1.0  # (Dx,)
        gy = (iy + 0.5) / self.Dy * 2.0 - 1.0  # (Dy,)

        # z in [0, 1] for Fourier PE
        gz = (iz + 0.5) / self.Dz  # (Dz,)  ∈ (0,1)

        # Build full grid: iterate all (ix, iy, iz)
        # Result order: (Dx, Dy, Dz) → flattened N
        gx_3d, gy_3d, gz_3d = torch.meshgrid(gx, gy, gz, indexing='ij')
        # (N, 2)  grid_sample want (x, y) pair
        grid_xy = torch.stack([gx_3d.reshape(-1), gy_3d.reshape(-1)], dim=-1)
        # (N, 1)
        grid_z = gz_3d.reshape(-1, 1)

        self.register_buffer('_infer_grid_xy', grid_xy)  # (N, 2)
        self.register_buffer('_infer_grid_z', grid_z)    # (N, 1)

    def _idx_to_coords(self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor):
        """Convert integer voxel indices to normalised sampling coordinates.

        Args:
            ix: (N,) long – voxel index along x
            iy: (N,) long – voxel index along y
            iz: (N,) long – voxel index along z

        Returns:
            grid_xy: (N, 2) float in [-1, 1]
            z_norm:  (N, 1) float in [0, 1]
        """
        gx = (ix.float() + 0.5) / self.Dx * 2.0 - 1.0
        gy = (iy.float() + 0.5) / self.Dy * 2.0 - 1.0
        z_norm = (iz.float() + 0.5) / self.Dz          # ∈ (0, 1)
        grid_xy = torch.stack([gx, gy], dim=-1)         # (N, 2)
        return grid_xy, z_norm.unsqueeze(-1)             # (N,2), (N,1)

    def _query_mlp(
        self,
        bev_feat: torch.Tensor,
        grid_xy: torch.Tensor,
        z_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Run bilinear BEV sampling + z-PE + MLP for a single sample.

        Args:
            bev_feat: (C, Dy, Dx)
            grid_xy:  (N, 2) in [-1, 1]
            z_norm:   (N, 1) in [0, 1]

        Returns:
            logits: (N, num_classes)
        """
        N = grid_xy.shape[0]
        # grid_sample: input (1,C,Dy,Dx), grid (1,1,N,2) → output (1,C,1,N)
        grid = grid_xy.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        feat = F.grid_sample(
            bev_feat.unsqueeze(0),
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )  # (1, C, 1, N)
        feat = feat.squeeze(0).squeeze(1).t()      # (N, C)

        # z positional encoding
        z_enc = self.z_pe(z_norm)                  # (N, z_feat_dim)

        feat_full = torch.cat([feat, z_enc], dim=-1)  # (N, C+z_feat_dim)
        logits = self.mlp(feat_full)                  # (N, num_classes)
        return logits

    # ------------------------------------------------------------------
    # Training: point-sampled forward + loss in one call
    # ------------------------------------------------------------------

    def forward_train_loss(
        self,
        bev_feat: torch.Tensor,
        voxel_semantics: torch.Tensor,
        mask_camera: torch.Tensor,
    ) -> dict:
        """Efficient training path: sample valid voxels, predict, compute loss.

        Args:
            bev_feat:        (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)  long, GT class per voxel
            mask_camera:     (B, Dx, Dy, Dz)  bool/int, 1=camera-visible

        Returns:
            loss_dict with 'loss_occ' (+ aux losses if use_aux_losses).
        """
        B = bev_feat.shape[0]
        device = bev_feat.device

        batch_logits = []
        batch_labels = []

        for b in range(B):
            if self.use_mask:
                valid = mask_camera[b].bool()        # (Dx, Dy, Dz)
            else:
                valid = torch.ones(
                    self.Dx, self.Dy, self.Dz, dtype=torch.bool, device=device
                )

            # Ignore voxels with label 255 (unlabelled)
            valid = valid & (voxel_semantics[b] != 255)

            pts_idx = valid.nonzero(as_tuple=False)  # (N_valid, 3): ix,iy,iz
            if pts_idx.shape[0] == 0:
                continue

            # Sub-sample if too many points
            if self.max_train_pts and pts_idx.shape[0] > self.max_train_pts:
                perm = torch.randperm(pts_idx.shape[0], device=device)
                pts_idx = pts_idx[perm[: self.max_train_pts]]

            ix, iy, iz = pts_idx[:, 0], pts_idx[:, 1], pts_idx[:, 2]
            grid_xy, z_norm = self._idx_to_coords(ix, iy, iz)

            logits = self._query_mlp(bev_feat[b], grid_xy, z_norm)  # (N, n_cls)
            labels = voxel_semantics[b][ix, iy, iz].long()           # (N,)

            batch_logits.append(logits)
            batch_labels.append(labels)

        if not batch_logits:
            # Degenerate: all masked out — return zero loss tied to params
            return {'loss_occ': bev_feat.sum() * 0.0}

        all_logits = torch.cat(batch_logits, dim=0)  # (M, n_cls)
        all_labels = torch.cat(batch_labels, dim=0)  # (M,)

        loss_dict = {}

        # Main cross-entropy loss
        if self.class_balance:
            w = self.cls_weights.to(all_logits.dtype)
            loss_ce = F.cross_entropy(all_logits, all_labels, weight=w)
        else:
            loss_ce = F.cross_entropy(all_logits, all_labels, ignore_index=255)
        loss_dict['loss_occ'] = loss_ce

        # Lovasz loss can operate directly on flattened probabilities [P, C] and labels [P]
        if self.use_aux_losses:
            probs = torch.softmax(all_logits, dim=-1)
            loss_dict['loss_lovasz'] = lovasz_softmax_flat(probs, all_labels)

        return loss_dict

    # ------------------------------------------------------------------
    # Inference: full-grid forward
    # ------------------------------------------------------------------

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        """Full-grid inference forward pass.

        Args:
            bev_feat: (B, C, Dy, Dx)

        Returns:
            occ_pred: (B, Dx, Dy, Dz, num_classes)
        """
        B, C, Dy, Dx = bev_feat.shape
        N = self.Dx * self.Dy * self.Dz
        device = bev_feat.device

        grid_xy = self._infer_grid_xy.to(device)  # (N, 2)
        grid_z = self._infer_grid_z.to(device)     # (N, 1)

        all_logits = []
        # Process in chunks to bound peak memory
        for start in range(0, N, self.infer_chunk):
            end = min(start + self.infer_chunk, N)
            xy_chunk = grid_xy[start:end]       # (k, 2)
            z_chunk = grid_z[start:end]          # (k, 1)

            # Process all batch items for this chunk together
            # BEV sampling: (B, C, Dy, Dx) with grid (B, 1, k, 2) → (B, C, 1, k)
            grid = xy_chunk.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, 2)
            feat = F.grid_sample(
                bev_feat,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=False,
            )  # (B, C, 1, k)
            feat = feat.squeeze(2).permute(0, 2, 1)  # (B, k, C)

            # z PE — same for all batch items
            z_enc = self.z_pe(z_chunk)                          # (k, z_feat_dim)
            z_enc = z_enc.unsqueeze(0).expand(B, -1, -1)        # (B, k, z_feat_dim)

            feat_full = torch.cat([feat, z_enc], dim=-1)         # (B, k, C+z_dim)
            chunk_logits = self.mlp(feat_full)                   # (B, k, n_cls)
            all_logits.append(chunk_logits)

        logits = torch.cat(all_logits, dim=1)  # (B, N, n_cls)
        occ_pred = logits.reshape(B, self.Dx, self.Dy, self.Dz, self.num_classes)
        return occ_pred

    # ------------------------------------------------------------------
    # Loss (called after full-grid forward during val/test hooks if needed)
    # ------------------------------------------------------------------

    def loss(
        self,
        occ_pred: torch.Tensor,
        voxel_semantics: torch.Tensor,
        mask_camera: torch.Tensor,
    ) -> dict:
        """Standard loss interface (called during validation forward).

        Args:
            occ_pred:        (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera:     (B, Dx, Dy, Dz)
        """
        loss_dict = {}
        voxel_semantics = voxel_semantics.long()

        # (B, n_cls, Dx, Dy, Dz) for lovasz/scal API
        preds_nchw = occ_pred.permute(0, 4, 1, 2, 3).contiguous()

        if self.use_mask:
            mask = mask_camera.to(torch.int32).reshape(-1)
            sem_flat = voxel_semantics.reshape(-1)
            pred_flat = occ_pred.reshape(-1, self.num_classes)
            num_total = mask.sum()

            if self.class_balance:
                loss_ce = self.loss_occ_fn(
                    pred_flat, sem_flat, mask, avg_factor=num_total
                )
            else:
                loss_ce = self.loss_occ_fn(
                    pred_flat, sem_flat, mask, avg_factor=num_total
                )
        else:
            pred_flat = occ_pred.reshape(-1, self.num_classes)
            sem_flat = voxel_semantics.reshape(-1)
            loss_ce = self.loss_occ_fn(pred_flat, sem_flat)

        loss_dict['loss_occ'] = loss_ce

        if self.use_aux_losses:
            loss_dict['loss_voxel_sem_scal'] = sem_scal_loss(
                preds_nchw, voxel_semantics
            )
            loss_dict['loss_voxel_geo_scal'] = geo_scal_loss(
                preds_nchw, voxel_semantics, non_empty_idx=17
            )
            loss_dict['loss_voxel_lovasz'] = lovasz_softmax(
                torch.softmax(preds_nchw, dim=1), voxel_semantics
            )

        return loss_dict

    # ------------------------------------------------------------------
    # Get predictions
    # ------------------------------------------------------------------

    def get_occ(self, occ_pred: torch.Tensor, img_metas=None):
        """
        Args:
            occ_pred: (B, Dx, Dy, Dz, n_cls)
        Returns:
            List[(Dx, Dy, Dz), ...]
        """
        occ_score = occ_pred.softmax(-1)
        occ_res = occ_score.argmax(-1).cpu().numpy().astype(np.uint8)
        return list(occ_res)
