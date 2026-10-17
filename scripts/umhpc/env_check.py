"""Sanity checks on a UMHPC A100 node: GPU, library versions, cached weights, rough DIRE query cost."""
import os
import sys
import time
from pathlib import Path

import torch

W = Path(os.environ["WEIGHTS_ROOT"])


def section(title):
    print(f"\n== {title} ==", flush=True)


section("GPU")
assert torch.cuda.is_available(), "CUDA not available - are you on a gpu-a100 node?"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
print(f"{p.name} | {p.total_memory / 2**30:.1f} GiB | sm_{p.major}{p.minor}")
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
print(f"fp32 matmul (TF32 {'on' if torch.backends.cuda.matmul.allow_tf32 else 'off'}): "
      f"{20 * 2 * 4096**3 / (time.time() - t) / 1e12:.1f} TFLOP/s")

section("FFT round trip (unitary, as in Eq. 2)")
img = torch.rand(3, 256, 256, device="cuda", dtype=torch.float64)
X = torch.fft.fft2(img, norm="ortho")
err = (torch.fft.ifft2(X, norm="ortho").real - img).abs().max().item()
print(f"max |ifft2(fft2(x)) - x| = {err:.2e}")

section("Python packages")
import lpips, foolbox, imagehash, mpi4py, numpy, guided_diffusion  # noqa: E401,F401
print("numpy", numpy.__version__, "| lpips", lpips.__version__ if hasattr(lpips, "__version__") else "ok",
      "| foolbox", foolbox.__version__, "| mpi4py", mpi4py.__version__)

section("Weights")
required = {
    "ADM lsun_bedroom": W / "guided_diffusion/lsun_bedroom.pt",
    "DIRE lsun_adm": W / "dire/lsun_adm.pth",
    "UFD fc head": W / "ufd/fc_weights.pth",
    "CLIP ViT-L/14": W / "clip/ViT-L-14.pt",
}
missing = [k for k, v in required.items() if not v.is_file()]
for k, v in required.items():
    print(f"{'OK ' if v.is_file() else 'MISSING'} {k}: {v}")

section("LPIPS (offline)")
loss = lpips.LPIPS(net="alex", version="0.1", verbose=False).cuda().eval()
a = torch.rand(1, 3, 256, 256, device="cuda")
print("LPIPS(x, x) =", loss(a * 2 - 1, a * 2 - 1).item())

section("ADM UNet speed (one DIRE query = 20 DDIM inversion + 20 DDIM sampling steps)")
if (W / "guided_diffusion/lsun_bedroom.pt").is_file():
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    cfg = model_and_diffusion_defaults()
    cfg.update(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, dropout=0.1,
               image_size=256, learn_sigma=True, noise_schedule="linear", num_channels=256,
               num_head_channels=64, num_res_blocks=2, resblock_updown=True, use_fp16=True,
               use_scale_shift_norm=True, timestep_respacing="ddim20")
    model, _ = create_model_and_diffusion(**cfg)
    model.load_state_dict(torch.load(W / "guided_diffusion/lsun_bedroom.pt", map_location="cpu"))
    model.convert_to_fp16()
    model.cuda().eval()
    for bs in (1, 8, 32):
        xb = torch.randn(bs, 3, 256, 256, device="cuda")
        tb = torch.full((bs,), 500, device="cuda", dtype=torch.long)
        with torch.no_grad():
            model(xb, tb)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10):
                model(xb, tb)
            torch.cuda.synchronize()
        per_step = (time.time() - t) / 10
        print(f"batch {bs:>2}: {per_step * 1e3:7.1f} ms/step -> ~{per_step * 40 / bs:.3f} s per DIRE query per image "
              f"| peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
else:
    print("skipped (lsun_bedroom.pt missing)")

if missing:
    print(f"\nFAILED: missing weights: {', '.join(missing)}")
    sys.exit(1)
print("\nAll checks passed.")
