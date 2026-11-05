"""Load Prithvi-EO-2.0 (trained or random-init), read token activations, splice layers.

The model code (prithvi_mae.py) ships with each checkpoint on Hugging Face and lives
in DATA_ROOT/models/<variant>/, next to config.json and the .pt file.

Token layout: forward_features returns one tensor per transformer block, shaped
(B, 1 + T*h*w, D). Index 0 is the CLS token; patch tokens follow row by row.
The last entry has the final LayerNorm applied.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from .bands import normalize
from .config import DATA_ROOT

MODEL_DIRS = {  # Hugging Face ibm-nasa-geospatial/Prithvi-EO-2.0-<name>
    "tiny-tl": DATA_ROOT / "models" / "tiny-tl",
    "100m-tl": DATA_ROOT / "models" / "100m-tl",
    "300m-tl": DATA_ROOT / "models" / "300m-tl",
    "300m": DATA_ROOT / "models" / "300m",
    "600m-tl": DATA_ROOT / "models" / "600m-tl",
    "600m": DATA_ROOT / "models" / "600m",
}

# Patch size of the model in use (16 px for tiny/100M/300M, 14 px for 600M). Scripts call
# set_patch(variant_patch(variant)) once; patch_spectra and valid_patches then use it by default.
PATCH = 16


def variant_patch(variant: str) -> int:
    cfg = json.loads((MODEL_DIRS[variant] / "config.json").read_text())["pretrained_cfg"]
    return int(cfg["patch_size"][-1])


def set_patch(patch: int):
    global PATCH
    PATCH = int(patch)


def _import_model_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location(f"prithvi_mae_{model_dir.name}", model_dir / "prithvi_mae.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prithvi(variant: str = "tiny-tl", num_frames: int = 1, random_init_seed: int | None = None,
                 device: str | None = None) -> torch.nn.Module:
    """Build PrithviMAE for `num_frames` dates. With random_init_seed set, keep the
    untrained weights (the architecture-only control); otherwise load the checkpoint.

    The fixed sin/cos position embeddings are rebuilt for num_frames, exactly as the
    official inference.py does, so single-date inference is supported.
    """
    model_dir = MODEL_DIRS[variant]
    cfg = json.loads((model_dir / "config.json").read_text())["pretrained_cfg"]
    module = _import_model_module(model_dir)
    if random_init_seed is not None:
        torch.manual_seed(random_init_seed)
    model = module.PrithviMAE(**{**cfg, "num_frames": num_frames})
    if random_init_seed is None:
        ckpt = next(model_dir.glob("*.pt"))
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state["encoder.pos_embed"] = model.encoder.pos_embed
        state["decoder.decoder_pos_embed"] = model.decoder.decoder_pos_embed
        model.load_state_dict(state, strict=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def model_inputs(reflectance: np.ndarray, device, dates=None, locations=None, normalizer=normalize):
    """Reflectance (N, 6, H, W) x 10,000 -> normalized tensor (N, 6, 1, H, W) plus
    optional coords. dates: (N, 2) [year, day-of-year].

    locations: (N, 2), passed through unchanged. OPEN ITEM: the official
    inference.py passes rasterio's lnglat(), i.e. (lon, lat), while the encoder's
    docstring says (lat, lon). Confirm against the pretraining data loader before
    any latitude-mirroring swap (step 32); until then, match inference.py.
    """
    x = torch.as_tensor(normalizer(reflectance), dtype=torch.float32, device=device).unsqueeze(2)
    t = None if dates is None else torch.as_tensor(np.asarray(dates, float), dtype=torch.float32,
                                                   device=device).reshape(-1, 1, 2)
    loc = None if locations is None else torch.as_tensor(np.asarray(locations, float),
                                                         dtype=torch.float32, device=device).reshape(-1, 2)
    return x, t, loc


@torch.no_grad()
def token_activations(model, reflectance: np.ndarray, layers: list[int], dates=None, locations=None,
                      batch_size: int = 16, normalizer=normalize) -> dict[int, np.ndarray]:
    """Patch-token activations at the given block indices: {layer: (N, tokens, D)}.
    The CLS token is dropped. Spectral edits must already be applied to `reflectance`.
    `normalizer` is Prithvi's own by default; others exist only for control experiments."""
    device = next(model.parameters()).device
    out = {layer: [] for layer in layers}
    for s in range(0, len(reflectance), batch_size):
        sl = slice(s, s + batch_size)
        x, t, loc = model_inputs(reflectance[sl], device,
                                 None if dates is None else np.asarray(dates)[sl],
                                 None if locations is None else np.asarray(locations)[sl], normalizer)
        feats = model.forward_features(x, t, loc)
        for layer in layers:
            out[layer].append(feats[layer][:, 1:, :].float().cpu().numpy())
    return {layer: np.concatenate(v) for layer, v in out.items()}


def token_feature_fn(model, layer: int, dates=None, locations=None, readout=None, batch_size: int = 16):
    """Wrap the model as a battery feature function: tiles (N, 6, H, W) -> (N*tokens, F).

    Each output row is one (tile, token) sample. readout maps (M, D) activations to
    (M, F) features; it defaults to all D neurons, and later it will be an SAE encoder.
    """
    def f(tiles: np.ndarray) -> np.ndarray:
        acts = token_activations(model, tiles, [layer], dates, locations, batch_size)[layer]
        flat = acts.reshape(-1, acts.shape[-1])
        return flat if readout is None else readout(flat)
    return f


def patch_spectra(tiles: np.ndarray, patch: int | None = None) -> np.ndarray:
    """Mean spectrum of every patch, in token order: (N, 6, H, W) -> (N*tokens, 6)."""
    patch = patch or PATCH
    n, c, h, w = tiles.shape
    p = tiles.reshape(n, c, h // patch, patch, w // patch, patch).mean(axis=(3, 5))
    return p.transpose(0, 2, 3, 1).reshape(-1, c)


def valid_patches(tiles: np.ndarray, patch: int | None = None, min_reflectance: float = 1.0) -> np.ndarray:
    """True for each token whose patch pixels all have every band >= min_reflectance.

    Ratios and log-scale edits are meaningless for non-positive reflectance, so
    mechanism tests use only valid tokens. The model still sees whole tiles.
    Returns (N*tokens,) in token order."""
    patch = patch or PATCH
    n, c, h, w = tiles.shape
    ok = (tiles >= min_reflectance).all(axis=1)  # (N, H, W)
    return ok.reshape(n, h // patch, patch, w // patch, patch).all(axis=(2, 4)).reshape(-1)


@contextlib.contextmanager
def splice(model, layer: int, fn):
    """Replace block `layer`'s output with fn(output) during the with-block.
    Used for the SAE's spliced loss (Stage 2) and for feature ablation (Stage 5)."""
    handle = model.encoder.blocks[layer].register_forward_hook(lambda _m, _inp, out: fn(out))
    try:
        yield
    finally:
        handle.remove()
