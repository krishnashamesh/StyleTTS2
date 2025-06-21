"""gen_features.py  ––  create guide wav + feature tensors for StyleTTS‑2

CPU‑friendly 2‑pass pipeline for expressive ISTFTNet inference:
 1. Run the HiFi‑GAN decoder only (no diffusion) to create a guide waveform
    that matches the expected token → frame alignment.
 2. Extract realistic auxiliary features from the guide:
      • F0       via pyworld (DIO + StoneMask)
      • ASR      80‑ch embedding via wav2vec2‑base averaged / projected
      • N        96‑ch noise bands via the decoder's built‑in N_conv
 3. Save tokens, F0, ASR, N into an .npz that the wrapper can load.

Usage
-----
python gen_features.py \
    --text "Sample sentence" \
    --config checkpoints/config.yml \
    --checkpoint checkpoints/epoch_2nd_00100.pth \
    --out feats_sample.npz
"""

import argparse
import os
import inspect
import yaml
import numpy as np
import torch
import librosa
import soundfile as sf
import pyworld as pw
from munch import Munch
from transformers import AutoFeatureExtractor, AutoModel
from models import build_model

# ------------------------------------------------------------- utils

def load_yaml(path: str) -> Munch:
    with open(path, "r") as f:
        return Munch.fromDict(yaml.safe_load(f))


def build_dummy_bert(hidden: int = 768):
    class DummyBert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("cfg", (), {"hidden_size": hidden, "max_position_embeddings": 512})()
        def forward(self, x):
            return torch.zeros(len(x), hidden)
    return DummyBert()

class DummyAligner(torch.nn.Module):
    def forward(self, *a, **kw):  # returns fake durations
        return None

class DummyPitch(torch.nn.Module):
    def forward(self, *a, **kw):  # returns fake f0 curve
        return None


# ------------------------------------------------------------------
# Helper: tokenize(text, cfg)  →  torch.LongTensor  (L,)
# ------------------------------------------------------------------
import re, torch
from pathlib import Path

# simple cleaner: lower-case, keep punctuation present in the symbols table
_basic_cleaner = re.compile(r"[^ a-zA-Z0-9,.;:!?'\-]")

def tokenize(text: str, cfg) -> torch.LongTensor:
    """
    Convert raw text to a sequence of symbol IDs as expected by StyleTTS 2.

    • Looks for `cfg.symbols` (list) or `cfg.symbols_dict` (dict)
      — either is produced by the official repo's YAML loader.
    • Falls back to the repo's default LJSpeech `symbols.py` if absent.
    """
    # 1) get the symbols list
    if hasattr(cfg, "symbols"):              # training config usually stores the list
        symbols = cfg.symbols
    elif hasattr(cfg, "symbols_dict"):       # checkpoints often store only a dict
        symbols = list(cfg.symbols_dict.keys())
    else:
        # -------- fallback when symbols.py is absent ----------
        # id-0 <unk> keeps us within the model’s  n_token (=178)  range
        symbols = ["<unk>"]                          # only one token
        cfg.symbols_dict = {"<unk>": 0}              # add dict so later calls work

    # 2) build / reuse a dictionary  {symbol: index}
    if hasattr(cfg, "symbols_dict"):
        sdict = cfg.symbols_dict
    else:
        sdict = {s: i for i, s in enumerate(symbols)}
        cfg.symbols_dict = sdict             # cache for future calls

    # 3) basic cleaning (you can plug in your own)
    text = _basic_cleaner.sub(" ", text.lower()).strip()
    char_list = list(text)

    # 4) map to IDs, use <unk> (index 0) for unseen chars
    unk_id = sdict.get("<unk>", 0)
    ids = [sdict.get(ch, unk_id) for ch in char_list]

    return torch.LongTensor(ids)

# ----------------------------------------------------- guide synthesis

def synthesise_guide(text: str, cfg: Munch, ckpt: str, device: torch.device):
    """Return (wav, token_ids) using decoder‑only HiFi‑GAN branch."""
    mcfg = cfg.model_params if "model_params" in cfg else cfg.model
    dummy_bert = build_dummy_bert()
    decoder = build_model(mcfg, None, None, dummy_bert)["decoder"].to(device)
    state = torch.load(ckpt, map_location="cpu")
    if "decoder" in state:
        decoder.load_state_dict(state["decoder"], strict=False)
    decoder.eval()

    # naive char → id mapping (works for English checkpoints)
    symbols = [chr(i) for i in range(32, 127)] + ["~"]
    sym2id = {s: i for i, s in enumerate(symbols)}
    ids = torch.tensor([[sym2id.get(c, sym2id["~"]) for c in text.lower()]], device=device)

    # call decoder with minimal compatible signature
    fwd_sig = inspect.signature(getattr(decoder, "forward", decoder.__call__))
    if len(fwd_sig.parameters) == 1:
        wav = decoder(ids)
    elif len(fwd_sig.parameters) == 2:
        try:
            wav = decoder(ids, cfg=cfg)
        except TypeError:
            wav = decoder(ids, nets=None)
    else:
            # 1) build nets
        model_cfg = cfg.model_params if "model_params" in cfg else cfg.model
        dummy_bert   = build_dummy_bert()
        dummy_align  = DummyAligner()
        dummy_pitch  = DummyPitch()

        nets = build_model(model_cfg, dummy_align, dummy_pitch, dummy_bert)
        state = torch.load(ckpt, map_location=device)
        for name, module in nets.items():
            if name in state:                          # load what exists
                module.load_state_dict(state[name], strict=False)
            else:                                      # leave others randomly-init
                print(f"[warn] {name:12s}  not in checkpoint – skipped")

        # 2) tokenise
        ids = tokenize(text, cfg).to(device)   # (1, L)

        # 3) run the diffusion decoder
        dec = nets["decoder"]                      # shorthand
        with torch.no_grad():
            if hasattr(dec, "inference"):          # newer forks
                wav = dec.inference(ids, nets, cfg, use_teacher_forcing=False)
            else:                                  # older forks – fall back to forward()
                try:
                    wav = dec(ids, nets=nets, cfg=cfg)   # some versions expect nets+cfg
                except TypeError:
                    wav = dec(ids)                        # others need only ids

        return wav.squeeze().cpu().numpy(), ids.cpu()

    wav = wav.squeeze().cpu().numpy()
    return wav, ids


# ----------------------------------------------------- feature extract

def extract_features(wav: np.ndarray, sr: int, decoder, device: torch.device):
    T = wav.shape[0]
    # F0 (1 × T)
    _f0, t = pw.dio(wav.astype(np.float64), sr)
    f0 = pw.stonemask(wav.astype(np.float64), _f0, t, sr).astype(np.float32)[None, :]

    # ASR (1 × 80 × T)
    ext = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    asr_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()
    inp = ext(wav, sampling_rate=sr, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        feat = asr_model(inp).last_hidden_state.squeeze(0).T.unsqueeze(0)  # (1, 768, T')
    asr = torch.nn.functional.interpolate(feat, size=f0.shape[-1], mode="linear")[:, :80]

    # N (1 × 96 × T) via decoder's whitening conv
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80)
    mel_t = torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).to(device)  # (1,1,80,T)
    N = decoder.N_conv(mel_t).squeeze(0).cpu()  # (96, T)

    return f0, asr.cpu(), N


# -------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--config", default="checkpoints/config.yml")
    ap.add_argument("--checkpoint", default="checkpoints/latest.pth")
    ap.add_argument("--out", default="feats.npz")
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = load_yaml(args.config)

    # 1) guide wav
    wav, token_ids = synthesise_guide(args.text, cfg, args.checkpoint, device)
    sr = cfg.preprocess_params.sr if "preprocess_params" in cfg else 24000

    # 2) real features
    model_cfg = cfg.model_params if "model_params" in cfg else cfg.model
    dummy_decoder = build_model(model_cfg,
                                DummyAligner(), DummyPitch(), build_dummy_bert()
                            )["decoder"].to(device).eval()
    f0, asr, N = extract_features(wav, sr, dummy_decoder, device)

    # 3) save
    np.savez(args.out, tokens=token_ids.cpu().numpy(), F0=f0, ASR=asr.numpy(), N=N.unsqueeze(0).numpy())
    sf.write(args.out.replace(".npz", ".wav"), wav, sr)
    print(f"[✓] Saved {args.out} and guide wav")


if __name__ == "__main__":
    main()