"""Deterministic label-only detector adapters on the 256x256 float [0, 1] attack canvas.

Each adapter exposes
    prob_fake(x)   -> (N,) float    (used only for clean screening diagnostics)
    predict_real(x)-> (N,) bool     (the only interface attacks see, via QueryOracle)

Third-party code is imported by file path from $THIRD_PARTY to avoid package-name clashes
(DIRE and UniversalFakeDetect both ship a top-level `networks` package).
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _third_party():
    return Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[1] / "third_party"))


def _weights():
    return Path(os.environ.get("WEIGHTS_ROOT", Path(__file__).resolve().parents[1] / "weights"))


def _load_package(name, init_file):
    spec = importlib.util.spec_from_file_location(name, init_file,
                                                  submodule_search_locations=[str(Path(init_file).parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(x, mean, std):
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


def center_crop(x, size):
    h, w = x.shape[-2:]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


class Detector:
    name = "base"
    threshold = 0.5

    def prob_fake(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def predict_real(self, x):
        return self.prob_fake(x) <= self.threshold

    def describe(self):
        return dict(name=self.name, threshold=self.threshold)


class UniversalFakeDetect(Detector):
    """Ojha et al. 2023: frozen CLIP ViT-L/14 image features + linear head (fc_weights.pth).

    Official test transform: CenterCrop(224) -> ToTensor -> Normalize(CLIP mean/std); no resize.
    """
    name = "ufd"

    def __init__(self, device, threshold=0.5, fc_path=None, clip_root=None):
        self.device, self.threshold = device, threshold
        clip_pkg = _load_package("ufd_clip", _third_party() / "UniversalFakeDetect/models/clip/__init__.py")
        clip_root = clip_root or os.environ.get("CLIP_CACHE", _weights() / "clip")
        model, _ = clip_pkg.load("ViT-L/14", device="cpu", jit=False, download_root=str(clip_root))
        self.clip = model.float().eval().to(device)
        self.fc = torch.nn.Linear(768, 1)
        fc_path = Path(fc_path or _weights() / "ufd/fc_weights.pth")
        self.fc.load_state_dict(torch.load(fc_path, map_location="cpu"))
        self.fc.eval().to(device)
        self.files = {"fc": (str(fc_path), sha256_file(fc_path)),
                      "clip": (str(Path(clip_root) / "ViT-L-14.pt"), sha256_file(Path(clip_root) / "ViT-L-14.pt"))}

    @torch.no_grad()
    def prob_fake(self, x):
        x = _normalize(center_crop(x.to(self.device, torch.float32), 224), CLIP_MEAN, CLIP_STD)
        return torch.sigmoid(self.fc(self.clip.encode_image(x).float())).flatten()

    def describe(self):
        return dict(name=self.name, threshold=self.threshold, files=self.files,
                    transform="center_crop_224 -> clip_normalize")


class DIRE(Detector):
    """Wang et al. 2023. Every call recomputes the DIRE map from the given RGB image:

        x in [0,1] -> 2x-1 -> DDIM inversion (ddim20) -> DDIM sampling (ddim20) -> |x - recon|
        -> uint8 PNG-equivalent quantisation (dire * 255 / 2, truncated) -> /255
        -> Resize(256) (no-op) -> CenterCrop(224) -> ImageNet normalisation -> ResNet-50 -> sigmoid

    matching guided-diffusion/compute_dire.py followed by demo.py. DDIM with eta=0 is deterministic.
    """
    name = "dire"

    def __init__(self, device, threshold=0.5, recon_model=None, classifier=None, use_fp16=True,
                 timestep_respacing="ddim20"):
        self.device, self.threshold = device, threshold
        gd_root = _third_party() / "DIRE/guided-diffusion"
        if str(gd_root) not in sys.path:
            sys.path.insert(0, str(gd_root))
        from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

        args = model_and_diffusion_defaults()
        args.update(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, dropout=0.1,
                    image_size=256, learn_sigma=True, noise_schedule="linear", num_channels=256,
                    num_head_channels=64, num_res_blocks=2, resblock_updown=True, use_fp16=use_fp16,
                    use_scale_shift_norm=True, timestep_respacing=timestep_respacing)
        self.recon_args = args
        model, self.diffusion = create_model_and_diffusion(**args)
        recon_model = Path(recon_model or _weights() / "guided_diffusion/lsun_bedroom.pt")
        model.load_state_dict(torch.load(recon_model, map_location="cpu"))
        if use_fp16:
            model.convert_to_fp16()
        self.unet = model.to(device).eval()

        spec = importlib.util.spec_from_file_location("dire_resnet", _third_party() / "DIRE/networks/resnet.py")
        resnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resnet)
        clf = resnet.resnet50(num_classes=1)
        classifier = Path(classifier or _weights() / "dire/lsun_adm.pth")
        state = torch.load(classifier, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        clf.load_state_dict(state)
        self.clf = clf.to(device).eval()
        self.files = {"recon": (str(recon_model), sha256_file(recon_model)),
                      "classifier": (str(classifier), sha256_file(classifier))}

    @torch.no_grad()
    def dire_map(self, x):
        imgs = x.to(self.device, torch.float32) * 2 - 1
        shape = tuple(imgs.shape)
        latent = self.diffusion.ddim_reverse_sample_loop(self.unet, shape, noise=imgs, clip_denoised=True,
                                                          model_kwargs={}, real_step=0)
        recons = self.diffusion.ddim_sample_loop(self.unet, shape, noise=latent, clip_denoised=True,
                                                 model_kwargs={}, real_step=0)
        dire = (imgs - recons).abs()
        return (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8).float() / 255.0

    @torch.no_grad()
    def prob_fake(self, x):
        d = _normalize(center_crop(self.dire_map(x), 224), IMAGENET_MEAN, IMAGENET_STD)
        return torch.sigmoid(self.clf(d)).flatten()

    def describe(self):
        return dict(name=self.name, threshold=self.threshold, files=self.files,
                    recon_args={k: v for k, v in self.recon_args.items()},
                    transform="2x-1 -> ddim20 inversion+sampling -> |x-recon| -> uint8 -> crop224 -> imagenet_norm")


class LinearToyDetector(Detector):
    """Deterministic toy detector for tests: fake iff <w, x - b> > 0."""
    name = "toy"

    def __init__(self, w, b):
        self.w, self.b = w, b

    @torch.no_grad()
    def prob_fake(self, x):
        s = ((x - self.b[None]) * self.w[None]).flatten(1).sum(1)
        return torch.sigmoid(s)


def build_detector(name, device, cfg):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = cfg["detectors"][name]
    if name == "ufd":
        return UniversalFakeDetect(device, threshold=d["threshold"])
    if name == "dire":
        return DIRE(device, threshold=d["threshold"], use_fp16=d.get("use_fp16", True),
                    timestep_respacing=d.get("timestep_respacing", "ddim20"))
    raise ValueError(name)
