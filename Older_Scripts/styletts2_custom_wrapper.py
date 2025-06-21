from __future__ import annotations
import argparse
import inspect
import os
import importlib.util

import torch
import torchaudio
import yaml
from munch import Munch

# ───────────────────────── CPU shim ─────────────────────────────
if not torch.cuda.is_available():
    _orig_to = torch.Tensor.to
    def _safe_to(self, *args, **kw):
        if args and isinstance(args[0], str) and args[0].startswith("cuda"):
            args = ("cpu",) + args[1:]
        return _orig_to(self, *args, **kw)
    torch.Tensor.to = _safe_to

# ───────────────────────── helper loaders ──────────────────────
from models import build_model  # repo module

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _extract_model_cfg(raw: dict) -> dict:
    for k in ("model_params", "model"):
        if k in raw and "decoder" in raw[k]:
            return raw[k]
    return raw  # flat root already has decoder

# ---- PL‑BERT loader ---------------------------------------------------------

def _load_plbert(pl_dir: str, device):
    """Return a PL‑BERT or fallback BERT model."""
    try:
        util_py = os.path.join(pl_dir, "util.py")
        ckpt = os.path.join(pl_dir, "step_1100000.t7")
        if os.path.isfile(util_py) and os.path.isfile(ckpt):
            spec = importlib.util.spec_from_file_location("pl_util", util_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            bert = mod.load_plbert(pl_dir)
            print(f"[PL‑BERT] loaded from {pl_dir}")
            return bert.to(device).eval()
    except Exception as e:
        print(f"[PL‑BERT] fallback to bert-base-uncased → {e}")
    from transformers import AutoModel
    return AutoModel.from_pretrained("bert-base-uncased").to(device).eval()

# ───────────────────── wrapper ─────────────────────────────────
class StyleTTS2Wrapper(torch.nn.Module):
    def __init__(self, nets, full_cfg: Munch, device):
        super().__init__()
        self.nets = torch.nn.ModuleDict(nets)
        self.cfg = full_cfg
        self.device = device
        self.segment = full_cfg.get("train", {}).get("segment_size", 1024)
        # build fallback symbol map if missing
        if not hasattr(self.cfg, "symbol_to_id"):
            syms = [chr(i) for i in range(32, 127)] + ["~"]
            self.cfg.symbol_to_id = {s: i for i, s in enumerate(syms)}

    # ---------- tokenizer ----------
    def tokenize(self, text: str) -> torch.Tensor:
        ids = [self.cfg.symbol_to_id.get(c, self.cfg.symbol_to_id["~"]) for c in text.lower()]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # ---------- decoder runner ----------
    def _run_decoder(self, tokens: torch.Tensor):
        dec = self.nets["decoder"]
        sig = inspect.signature(getattr(dec, "forward", dec.__call__))
        seq_len = tokens.size(1)
        kw = {}

        # try to obtain F0 stride if present
        stride = 1
        if hasattr(dec, "F0_conv") and hasattr(dec.F0_conv, "stride"):
            stride = int(dec.F0_conv.stride[0]) or 1

        for p in list(sig.parameters.values())[1:]:  # skip self
            n = p.name.lower()
            if n in ("x", "tokens"):
                kw[p.name] = tokens
            elif n in ("n", "n_in"):
                kw[p.name] = self._aux.get("N") if hasattr(self, "_aux") else torch.zeros(1, seq_len, dtype=torch.float32, device=self.device)
            elif n in ("sid", "s", "spk", "speaker"):
                kw[p.name] = torch.zeros(1, dtype=torch.long, device=self.device)
            elif "f0" in n or "pitch" in n:
                kw[p.name] = self._aux.get("f0") if hasattr(self, "_aux") else torch.zeros(1, seq_len, dtype=torch.float32, device=self.device)
            elif "asr" in n:
                kw[p.name] = self._aux.get("asr") if hasattr(self, "_aux") else torch.zeros(1, 80, seq_len, dtype=torch.float32, device=self.device)
            elif n == "nets":
                kw[p.name] = self.nets
            elif n == "cfg":
                kw[p.name] = self.cfg
                # ensure mandatory tensors for F0 and ASR only (N handled above)
        if any("f0" in p.name.lower() or "pitch" in p.name.lower() for p in sig.parameters.values()):
            f0_key = next(p.name for p in sig.parameters.values() if "f0" in p.name.lower() or "pitch" in p.name.lower())
            kw.setdefault(f0_key, torch.zeros(1, seq_len, dtype=torch.float32, device=self.device))
        if any("asr" in p.name.lower() for p in sig.parameters.values()):
            asr_key = next(p.name for p in sig.parameters.values() if "asr" in p.name.lower())
            kw.setdefault(asr_key, torch.zeros(1, 80, seq_len, dtype=torch.float32, device=self.device))
        return dec(**kw)

    # ---------- diffusion + vocoder ----------
    @torch.inference_mode()
    def forward(self, text: str):
        tokens = self._aux["tokens"] if hasattr(self, "_aux") else self.tokenize(text).unsqueeze(0)
        mel_coarse = self._run_decoder(tokens)
        parts = [mel_coarse[:, :, i:i+self.segment] for i in range(0, mel_coarse.size(2), self.segment)]
        refined = []
        for chunk in parts:
            noise = torch.randn_like(chunk)
            refined.append(self.nets["diffusion"](noise, embedding=None, features=chunk))
        mel = torch.cat(refined, dim=2)
        return self.nets["vocoder"](mel)
    
    def set_aux_feats(self, feats_npz):
        import numpy as np, torch
        self._aux = {
            "tokens": torch.from_numpy(feats_npz["tokens"]).long().to(self.device),
            "f0":     torch.from_numpy(feats_npz["F0"]).float().to(self.device),
            "asr":    torch.from_numpy(feats_npz["ASR"]).float().to(self.device),
            "N":      torch.from_numpy(feats_npz["N"]).float().to(self.device),
        }


# ───────────────────────── main ───────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--config", default="checkpoints/config.yml")
    ap.add_argument("--checkpoint", default="checkpoints/latest.pth")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feats", required=True,
                help="Path to .npz from gen_features.py")
    return ap.parse_args()


def main():
    args = _parse_args()
    device = torch.device(args.device)

    raw_cfg = _load_yaml(args.config)
    full_cfg = Munch.fromDict(raw_cfg)
    model_cfg = Munch.fromDict(_extract_model_cfg(raw_cfg))

    bert = _load_plbert(full_cfg.get("PLBERT_dir", "Utils/PLBERT"), device)

    nets = build_model(model_cfg, None, None, bert)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    for name, net in nets.items():
        if net is None:
            continue
        net.load_state_dict(ckpt.get(name, {}), strict=False)
        net.to(device).eval()

    wrapper = StyleTTS2Wrapper(nets, full_cfg, device).to(device)
    import numpy as np
    feats = np.load(args.feats)
    wrapper.set_aux_feats(feats)        # new helper
    wav = wrapper(args.text).cpu()
    sr = full_cfg.get("preprocess_params", {}).get("sr", 24000)
    torchaudio.save(args.out, wav, sr)
    print(f"[✓] Saved → {args.out} ({wav.shape[-1]/sr:.2f}s)")

if __name__ == "__main__":
    main()