"""
StyleTTS2 *unified* runner with verbose logging
---------------------------------------------
• Decoder‑agnostic (ISTFTNet *or* DiffusionDecoder)
• CPU shim, flexible YAML, PL‑BERT fallback, chunked diffusion
• Now: **smarter PL‑BERT loader** — accepts any *.t7* in *Utils/PLBERT/*
"""
from __future__ import annotations
import argparse, inspect, importlib.util, os, sys, logging, glob
from types import ModuleType

import torch, torchaudio, yaml
from munch import Munch

################################################################################
# ───────────────────────────── logging setup ────────────────────────────────
################################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

################################################################################
# ──────────────────────────── CUDA → CPU shim ───────────────────────────────
################################################################################
if not torch.cuda.is_available():
    log.info("CUDA not available → patching torch.Tensor.to for CPU fallback")
    _orig_to = torch.Tensor.to
    def _safe_to(self, *args, **kw):
        if args and isinstance(args[0], str) and args[0].startswith("cuda"):
            args = ("cpu",) + args[1:]
        return _orig_to(self, *args, **kw)
    torch.Tensor.to = _safe_to  # type: ignore
else:
    log.info("CUDA detected – using GPU")

################################################################################
# ───────────────────────────── helpers / loaders ────────────────────────────
################################################################################
from models import build_model  # repo’s factory

# -- tiny recursive_munch fallback ------------------------------------------------
try:
    from models import recursive_munch  # type: ignore
except (ImportError, AttributeError):
    def recursive_munch(obj):
        if isinstance(obj, dict):
            return Munch({k: recursive_munch(v) for k, v in obj.items()})
        if isinstance(obj, (list, tuple)):
            return type(obj)(recursive_munch(x) for x in obj)
        return obj

# -----------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    log.info(f"Loading YAML config → {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_model_cfg(raw: dict) -> dict:
    for k in ("model_params", "model"):
        if k in raw and "decoder" in raw[k]:
            log.info(f"Model sub‑config found under '{k}' key")
            return raw[k]
    log.info("Using flat root for model config")
    return raw

# ---- PL‑BERT loader ---------------------------------------------------------

def _load_plbert(pl_dir: str, device):
    """Load PL‑BERT if *any* .t7 checkpoint exists in *pl_dir*; else fallback."""
    util_py = os.path.join(pl_dir, "util.py")
    ckpts = glob.glob(os.path.join(pl_dir, "*.t7"))
    if not os.path.isfile(util_py) or not ckpts:
        log.warning("PL‑BERT util.py or .t7 not found → falling back to bert‑base‑uncased")
        from transformers import AutoModel
        return AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

    ckpt_path = sorted(ckpts)[-1]  # pick highest step number
    log.info(f"Loading PL‑BERT ({os.path.basename(ckpt_path)}) from {pl_dir}")
    try:
        spec = importlib.util.spec_from_file_location("pl_util", util_py)
        mod: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore

        fn = getattr(mod, "load_plbert")
        if len(inspect.signature(fn).parameters) == 1:
            bert = fn(pl_dir)
        else:
            bert = fn(pl_dir, ckpt_path)
        return bert.to(device).eval()
    except Exception as e:
        log.warning(f"PL‑BERT load failed ({e}) → bert‑base‑uncased fallback")
        from transformers import AutoModel
        return AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

################################################################################
# ──────────────────────────── Wrapper Module ────────────────────────────────
################################################################################
class StyleTTS2Wrapper(torch.nn.Module):
    def __init__(self, nets, cfg: Munch, device: torch.device):
        super().__init__()
        self.nets = torch.nn.ModuleDict({k: v for k, v in nets.items() if v is not None})
        self.cfg = cfg
        self.device = device

        if not hasattr(self.cfg, "symbol_to_id"):
            syms = [chr(i) for i in range(32, 127)] + ["~"]
            self.cfg.symbol_to_id = {s: i for i, s in enumerate(syms)}
            log.info("Built fallback symbol_to_id map (ASCII + '~')")

        self.segment = (
            cfg.get("train", {}).get("segment_size") or
            cfg.get("dataset_params", {}).get("segment_size") or 1024
        )
        log.info(f"Segment size set to {self.segment}")

    # ---------- tokenizer ----------------------------------------------------
    def tokenize(self, text: str) -> torch.Tensor:
        ids = [self.cfg.symbol_to_id.get(c, self.cfg.symbol_to_id["~"]) for c in text.lower()]
        log.info(f"Tokenized '{text}' → {ids}")
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    # ---------- generic decoder runner --------------------------------------
    def _run_decoder(self, tokens: torch.Tensor):
        dec = self.nets["decoder"]
        sig = inspect.signature(getattr(dec, "forward", dec.__call__))
        log.info(f"Decoder type: {dec.__class__.__name__}")
        kwargs = {}
        seq_len = tokens.shape[1]
        mel_len = seq_len * 4  # heuristic for ISTFTNet dummy tensors
        log.info(f"Token len={seq_len} → mel len guess={mel_len}")

        def _maybe(name: str, shape):
            if any(name in p.name.lower() for p in sig.parameters.values()):
                kwargs[next(p.name for p in sig.parameters.values() if name in p.name.lower())] = (
                    torch.zeros(shape, device=self.device))

        for p in list(sig.parameters.values())[1:]:
            n = p.name.lower()
            if n in ("x", "tokens"):
                kwargs[p.name] = tokens
            elif n in ("sid", "spk", "speaker", "s"):
                kwargs[p.name] = torch.zeros(1, dtype=torch.long, device=self.device)
            elif n == "nets":
                kwargs[p.name] = self.nets
            elif n == "cfg":
                kwargs[p.name] = self.cfg
        
        # --- dummy tensors tuned for ISTFTNet --------------------------------
        asr_shape = (1, 512, mel_len)        # matches Decoder expectation
        f0n_shape = (1, mel_len * 2)         # will be halved by stride-2

        kwargs.setdefault("asr", torch.zeros(asr_shape, device=self.device))
        kwargs.setdefault("F0" , torch.zeros(f0n_shape,  device=self.device))
        kwargs.setdefault("N"  , torch.zeros(f0n_shape,  device=self.device))


        return dec(**kwargs)

    @torch.inference_mode()
    def forward(self, text: str):
        log.info(f"Synthesising: '{text}'")
        tokens = self.tokenize(text)
        mel_or_coarse = self._run_decoder(tokens)
        log.info(f"Decoder output shape: {tuple(mel_or_coarse.shape)}")

        if "diffusion" not in self.nets or self.nets["diffusion"] is None:
            log.info("No separate diffusion network → assuming full‑diffusion decoder")
            mel = mel_or_coarse
        else:
            log.info("Refining with separate diffusion network chunk‑wise …")
            chunks = [
                mel_or_coarse[:, :, i : i + self.segment]
                for i in range(0, mel_or_coarse.size(2), self.segment)
            ]
            refined = []
            for idx, c in enumerate(chunks):
                log.info(f"Diffusion chunk {idx+1}/{len(chunks)} shape={tuple(c.shape)}")
                noise = torch.randn_like(c)
                refined.append(self.nets["diffusion"](noise, embedding=None, features=c))
            mel = torch.cat(refined, dim=2)
            log.info("Diffusion refinement complete")

        if "vocoder" not in self.nets or self.nets["vocoder"] is None:
            raise RuntimeError("No vocoder present in checkpoint")
        log.info("Running vocoder …")
        return self.nets["vocoder"](mel)

################################################################################
# ─────────────────────────────── CLI & main ─────────────────────────────────
################################################################################

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--config", default="checkpoints/config.yml")
    ap.add_argument("--checkpoint", default="checkpoints/epoch_2nd_00100.pth")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feats", help="Optional .npz from gen_features.py")

    return ap.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device → {device}")

    raw_cfg = _load_yaml(args.config)
    full_cfg = recursive_munch(raw_cfg)
    model_cfg = recursive_munch(_extract_model_cfg(raw_cfg))

    bert = _load_plbert(full_cfg.get("PLBERT_dir", "Utils/PLBERT"), device)

    log.info("Building model …")
    nets = build_model(model_cfg, None, None, bert)

    log.info(f"Loading checkpoint → {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    for name, net in nets.items():
        if net is None:
            continue
        missing, unexpected = net.load_state_dict(ckpt.get(name, {}), strict=False)
        log.info(f"[{name}] missing={len(missing)} unexpected={len(unexpected)}")
    
    
    wrapper = StyleTTS2Wrapper(nets, full_cfg, device).to(device)

    if args.feats:
        import numpy as np
        feats = np.load(args.feats)
        wrapper.set_aux_feats(feats)

    wav = wrapper(args.text).cpu()
    sr = full_cfg.get("preprocess_params", {}).get("sr", 24000)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torchaudio.save(args.out, wav, sr)
    print(f"[✓] saved → {args.out}  ({wav.shape[-1]/sr:.2f}s, sr={sr})")

if __name__ == "__main__":
    main()