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


class AEROBLADE(Detector):
    """Ricker et al., CVPR 2024 (training-free), following github.com/jonasricker/aeroblade @ 50fc2b2.

    For each latent-diffusion autoencoder (SD1 VAE, SD2 VAE, Kandinsky 2.1 MoVQ):
        x in [0,1] -> 2x-1 -> encode -> latent sample (seed 1) -> decode -> /2+0.5 -> clip
        -> uint8 quantisation (the official code stores reconstructions as PNG)
        -> distance = spatial mean of LPIPS-VGG(v0.1) layer 2 (the official "lpips_vgg_2")
    score = max over autoencoders of (-distance); a small reconstruction error means "generated".
    SD VAEs use fp16 weights computed in fp32 (official upcast_vae); MoVQ runs in fp16.

    Departures needed for a label-only oracle: the latent noise is drawn from a fresh seed-1 generator
    for every image (official: one generator per directory, so noise depended on file order), and the
    forward pass runs without cuDNN and with classic attention, making the score a function of the
    image alone (independent of batch position); AEROBLADE defines no decision threshold, so `threshold`
    (in score units) is calibrated once on the development split (experiments/calibrate_threshold.py)
    and frozen before screening. prob_fake returns the raw score; Real iff score <= threshold.
    """
    name = "aeroblade"

    def __init__(self, device, threshold, autoencoders, lpips_layer=2):
        from diffusers import AutoencoderKL, VQModel
        import lpips

        self.device, self.threshold, self.layer = device, threshold, lpips_layer
        self.aes, self.files = [], {}
        for spec in autoencoders:
            kw = dict(revision=spec["revision"], torch_dtype=torch.float16, use_safetensors=True)
            if spec.get("subfolder"):
                kw["subfolder"] = spec["subfolder"]
            if spec.get("variant"):
                kw["variant"] = spec["variant"]
            if spec["type"] == "vq":
                ae = VQModel.from_pretrained(spec["repo"], **kw).half()
            else:
                ae = AutoencoderKL.from_pretrained(spec["repo"], **kw).half().float()
            # Classic (bmm) attention instead of fused SDPA kernels, so the computation does not depend on
            # which fused kernel a given GPU / batch shape selects.
            from diffusers.models.attention_processor import Attention, AttnProcessor
            for mod in ae.modules():
                if isinstance(mod, Attention):
                    mod.set_processor(AttnProcessor())
            self.aes.append(ae.to(device).eval())
            self.files[spec["repo"]] = (spec["repo"], f"{spec['revision']}:{_state_sha256(ae)}")
        self._vq = VQModel
        self.lpips = lpips.LPIPS(net="vgg", version="0.1", spatial=True, verbose=False).to(device).eval()
        self.autoencoders = autoencoders

    @torch.no_grad()
    def _distance(self, a, b):
        lp, k = self.lpips, self.layer - 1          # layer 2 = second VGG stage (index 1)
        fa, fb = lp.net.forward(lp.scaling_layer(2 * a - 1)), lp.net.forward(lp.scaling_layer(2 * b - 1))
        import lpips
        d = (lpips.normalize_tensor(fa[k]) - lpips.normalize_tensor(fb[k])) ** 2
        return lp.lins[k](d).mean((1, 2, 3))

    @torch.no_grad()
    def prob_fake(self, x):
        # cuDNN off: its fp32 strided-convolution kernels made a VAE output depend on the image's position in
        # the batch (~1e-6 relative, measured), so the label would not be a function of the image alone.
        with torch.backends.cudnn.flags(enabled=False):
            return self._score(x)

    def _score(self, x):
        x = x.to(self.device, torch.float32)
        scores = []
        for ae in self.aes:
            enc = ae.encode((x * 2 - 1).to(ae.dtype))
            if isinstance(ae, self._vq):
                rec = ae.decode(enc.latents, force_not_quantize=True).sample
            else:
                dist = enc.latent_dist
                g = torch.Generator().manual_seed(1)
                eps = torch.randn(dist.mean.shape[1:], generator=g).to(self.device, dist.mean.dtype)
                rec = ae.decode(dist.mean + dist.std * eps[None]).sample
            rec = ((rec.float() / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).float() / 255
            scores.append(-self._distance(x, rec))
        return torch.stack(scores).max(0).values

    def describe(self):
        return dict(name=self.name, threshold=self.threshold, files=self.files, lpips_layer=self.layer,
                    autoencoders=self.autoencoders, score="max_AE(-LPIPS_vgg_2(x, AE(x))); Real iff score <= threshold",
                    transform="2x-1 -> AE encode/sample(seed 1)/decode -> uint8 -> LPIPS vgg layer 2 spatial mean")


def _state_sha256(module):
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def load_calibrated_threshold(name, results):
    """Threshold frozen by experiments/calibrate_threshold.py (development split, never test)."""
    import json

    path = Path(results) / "calibration" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run experiments/calibrate_threshold.py --detector {name} first")
    return json.loads(path.read_text())["threshold"]


class LinearToyDetector(Detector):
    """Deterministic toy detector for tests: fake iff <w, x - b> > 0."""
    name = "toy"

    def __init__(self, w, b):
        self.w, self.b = w, b

    @torch.no_grad()
    def prob_fake(self, x):
        s = ((x - self.b[None]) * self.w[None]).flatten(1).sum(1)
        return torch.sigmoid(s)


def fixed_batch(fn, size):
    """Wrap a detector function so it always runs on chunks of exactly `size` images (zero-padded).

    The scheduler mixes queries from many attacks, so the number of images per call varies; kernel
    selection can depend on the batch shape, and near the decision boundary that numerical noise could
    flip a label. A fixed shape makes every image's label a deterministic function of the image alone.
    """
    @torch.no_grad()
    def wrapped(x):
        out = []
        for i in range(0, len(x), size):
            xi = x[i:i + size]
            k = len(xi)
            if k < size:
                xi = torch.cat([xi, xi.new_zeros((size - k,) + tuple(xi.shape[1:]))])
            out.append(fn(xi)[:k])
        return torch.cat(out)
    return wrapped


def build_detector(name, device, cfg, results=None, threshold=None):
    """threshold: override (calibration only); a "calibrated" config threshold is read from results/calibration."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = cfg["detectors"][name]
    if name == "aeroblade":
        if threshold is None:
            threshold = d["threshold"]
            if threshold == "calibrated":
                threshold = load_calibrated_threshold(name, results or "results")
        return AEROBLADE(device, threshold, d["autoencoders"], d.get("lpips_layer", 2))
    if name == "ufd":
        return UniversalFakeDetect(device, threshold=d["threshold"])
    if name == "dire":
        return DIRE(device, threshold=d["threshold"], use_fp16=d.get("use_fp16", True),
                    timestep_respacing=d.get("timestep_respacing", "ddim20"))
    raise ValueError(name)
