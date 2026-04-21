"""
Depth-estimation loss functions.

All losses accept `(pred, gt, mask)`:
    pred : (B, H, W) or (B, 1, H, W) predicted depth in metres (>0).
    gt   : (B, H, W) or (B, 1, H, W) ground-truth depth in metres.
    mask : (B, H, W) or (B, 1, H, W) boolean/float mask of valid pixels.

All losses gracefully handle an *empty* mask (no valid pixels in the batch) by
returning a zero tensor that still participates in the autograd graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _squeeze_chan(x: torch.Tensor) -> torch.Tensor:
    """Accept (B,H,W) or (B,1,H,W) and return (B,H,W)."""
    if x.dim() == 4 and x.size(1) == 1:
        x = x.squeeze(1)
    return x


def _zero_like(ref: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero tensor attached to `ref`'s graph."""
    return (ref * 0).sum()


# ----------------------------------------------------------------------
# 1. Scale-invariant (SILog) loss
# ----------------------------------------------------------------------
class ScaleInvariantLoss(nn.Module):
    """BTS-style Scale-Invariant log loss.

        d = log(pred) - log(gt)
        L = 10 * sqrt( mean(d^2) - lambda * (mean(d))^2 )
    """

    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-7):
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = _squeeze_chan(pred)
        gt = _squeeze_chan(gt)
        mask = _squeeze_chan(mask).bool()

        if mask.sum() == 0:
            return _zero_like(pred)

        p = pred[mask].clamp(min=self.eps)
        g = gt[mask].clamp(min=self.eps)
        d = torch.log(p) - torch.log(g)

        var_term = (d ** 2).mean() - self.variance_focus * (d.mean() ** 2)
        # guard against numerical negative values
        var_term = torch.clamp(var_term, min=self.eps)
        return 10.0 * torch.sqrt(var_term)


# ----------------------------------------------------------------------
# 2. L1 (log-space) depth loss
# ----------------------------------------------------------------------
class L1DepthLoss(nn.Module):
    """Mean absolute error in log space on valid pixels."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = _squeeze_chan(pred)
        gt = _squeeze_chan(gt)
        mask = _squeeze_chan(mask).bool()

        if mask.sum() == 0:
            return _zero_like(pred)

        p = torch.log(pred[mask].clamp(min=self.eps))
        g = torch.log(gt[mask].clamp(min=self.eps))
        return (p - g).abs().mean()


# ----------------------------------------------------------------------
# 3. Gradient-matching loss
# ----------------------------------------------------------------------
class GradientMatchingLoss(nn.Module):
    """Masked L1 on the x/y first-order spatial gradients of log depth."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _grad_xy(x: torch.Tensor):
        # x : (B, H, W)
        gx = x[:, :, 1:] - x[:, :, :-1]
        gy = x[:, 1:, :] - x[:, :-1, :]
        return gx, gy

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pred = _squeeze_chan(pred)
        gt = _squeeze_chan(gt)
        mask = _squeeze_chan(mask).float()

        if mask.sum() == 0:
            return _zero_like(pred)

        log_p = torch.log(pred.clamp(min=self.eps))
        log_g = torch.log(gt.clamp(min=self.eps))

        px, py = self._grad_xy(log_p)
        gx, gy = self._grad_xy(log_g)

        # gradient is valid only where both neighbours are valid.
        mx = mask[:, :, 1:] * mask[:, :, :-1]
        my = mask[:, 1:, :] * mask[:, :-1, :]

        loss = _zero_like(pred)
        n = 0.0

        if mx.sum() > 0:
            loss = loss + ((px - gx).abs() * mx).sum() / mx.sum().clamp(min=1.0)
            n += 1.0
        if my.sum() > 0:
            loss = loss + ((py - gy).abs() * my).sum() / my.sum().clamp(min=1.0)
            n += 1.0

        if n == 0:
            return _zero_like(pred)
        return loss / n


# ----------------------------------------------------------------------
# 4. Weighted combined loss
# ----------------------------------------------------------------------
class CombinedDepthLoss(nn.Module):
    """Weighted sum of SILog + L1 (log) + gradient-matching."""

    def __init__(
        self,
        silog_weight: float = 1.0,
        l1_weight: float = 0.1,
        grad_weight: float = 0.5,
        silog_variance_focus: float = 0.85,
    ):
        super().__init__()
        self.silog_weight = silog_weight
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight

        self.silog = ScaleInvariantLoss(variance_focus=silog_variance_focus)
        self.l1 = L1DepthLoss()
        self.grad = GradientMatchingLoss()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        l_silog = self.silog(pred, gt, mask)
        l_l1 = self.l1(pred, gt, mask)
        l_grad = self.grad(pred, gt, mask)

        total = (
            self.silog_weight * l_silog
            + self.l1_weight * l_l1
            + self.grad_weight * l_grad
        )
        return {
            "total": total,
            "silog": l_silog.detach(),
            "l1": l_l1.detach(),
            "grad": l_grad.detach(),
        }
