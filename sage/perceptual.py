"""Perceptual distance d_p (LPIPS AlexNet v0.1 on 2x-1) and diagnostic distortions (PSNR, RMSE)."""
import torch


class LPIPSDistance:
    """Frozen LPIPS. Call with (N,3,H,W) and (N,3,H,W) or (1,3,H,W) tensors in [0, 1]."""

    def __init__(self, device, net="alex", version="0.1"):
        import lpips

        self.model = lpips.LPIPS(net=net, version=version, verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.name = f"lpips-{net}-v{version}"

    @torch.no_grad()
    def __call__(self, x, ref, chunk=64):
        x = x.to(self.device, torch.float32)
        ref = ref.to(self.device, torch.float32)
        if ref.dim() == 3:
            ref = ref[None]
        out = []
        for i in range(0, len(x), chunk):
            xi = x[i:i + chunk]
            ri = ref.expand(len(xi), -1, -1, -1) if len(ref) == 1 else ref[i:i + chunk]
            out.append(self.model(xi * 2 - 1, ri * 2 - 1).flatten())
        return torch.cat(out)


def l2(x, ref):
    return (x.to(torch.float64) - ref.to(torch.float64)).flatten(1).norm(dim=1)


def rmse(x, ref):
    return (x.to(torch.float64) - ref.to(torch.float64)).flatten(1).pow(2).mean(1).sqrt()


def psnr(x, ref):
    mse = (x.to(torch.float64) - ref.to(torch.float64)).flatten(1).pow(2).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-20))
