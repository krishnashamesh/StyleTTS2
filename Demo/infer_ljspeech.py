#!/usr/bin/env python3
# Demo/infer_ljspeech.py  –  StyleTTS-2 inference (no-pad, clean fallback)
import os, sys, yaml, torch, torchaudio
sys.path.insert(0, "/opt/apps/StyleTTS2")

from munch import Munch
from models import build_model

# ──────────────────────────────────────────────────────────────────────────
# 1. tiny YAML→Munch helper
def yaml_to_munch(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    def _conv(obj):
        if isinstance(obj, dict):  return Munch({k: _conv(v) for k,v in obj.items()})
        if isinstance(obj, list):  return [_conv(v) for v in obj]
        return obj
    return _conv(raw)

# ──────────────────────────────────────────────────────────────────────────
# 2.  Wrapper with graceful style-encoder fallback
class StyleTTS2Wrapper(torch.nn.Module):
    def __init__(self, nets, cfg):
        super().__init__()
        self.nets   = torch.nn.ModuleDict(nets)
        self.cfg    = cfg
        # discover UNet hidden-channels once and build a matching projection
        # inside StyleTTS2Wrapper.__init__
        unet = nets.diffusion.unet
        self.u_ch = 1024

        self.proj = torch.nn.Linear(1458, self.u_ch)

    # simple ASCII tokenizer (truncates at 512 tokens)
    @staticmethod
    def _tok(txt):
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-"
        table = {c:i+1 for i,c in enumerate(vocab)}
        ids   = [table.get(c,0) for c in txt.lower()][:512]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    # ──────────────────────────────────────────────────────────────
    #  forward – clean shapes, no-pad, 1-channel diffusion noise
    def forward(self, text: str) -> torch.Tensor:
        dev  = next(self.parameters()).device

        # --- basic token / mask ------------------------------------------------
        toks = self._tok(text).to(dev)              # [1,T]
        T    = toks.shape[1]
        lens = torch.tensor([T], device=dev)
        mask = torch.zeros(1, T, dtype=torch.bool, device=dev)  # [1,T]

        # --- 1) text encoder ---------------------------------------------------
        h_txt = self.nets.text_encoder(toks, lens, mask)        # [1,512,T]

        # --- 2) style encoder (requires height ≥80, width divisible by 16) -----
        if T >= 80:
            T16  = (T // 16) * 16           # largest multiple of 16 ≤ T
            style_in = h_txt[:, :80, :T16]  # use first 80 channels as “mel”
            style_in = style_in.unsqueeze(1)             # [1,1,80,T16]
            h_sty = self.nets.style_encoder(style_in)    # [1,128]
            h_txt = h_txt[:, :, :T16]        # keep same length for prosody
            T_use = T16
        else:                                # text shorter than 80 frames
            h_sty = torch.zeros(1, self.cfg.style_dim, device=dev)
            T_use = T

        # --- 3) prosody predictor (duration + energy) --------------------------
        eye = torch.eye(T_use, device=dev).unsqueeze(0)          # [1,T,T]
        dur, en = self.nets.predictor(
                      h_txt, h_sty, torch.tensor([T_use], device=dev),
                      eye, mask[:, :T_use])                      # [1,T,50] [1,T,640]

        # energy comes out as [B, 640, T]  →  make it [B, T, 640]
        en   = en.transpose(1, 2)                    # [1, T_use, 640]
        pitch = torch.zeros_like(en)                 # [1, T_use, 640]

        # duration already [1, T_use, 50]
        pros  = torch.cat((dur, pitch, en), dim=-1)  # [1, T_use, 1330]

        # conditioning: prosody + style  → project to UNet-channels
        cond = torch.cat((pros, h_sty.unsqueeze(1).expand(-1, T_use, -1)), -1)  # [1,T,1458]
        cond = self.proj(cond)                                   # [1,T,u_ch]

        # --- 4) BERT embedding (layout [B,T,768]) -------------------------------
        bert = self.nets.bert(toks)[0]                           # [1,T,768] or [T,768]
        if bert.dim() == 2:
            bert = bert.unsqueeze(0)
        bert = bert[:, :T_use, :]                                # [1,T_use,768]

        # --- 5) diffusion ------------------------------------------------------
        noise = torch.randn(1, 1, T_use, device=dev)             # **1-channel**
        mel   = self.nets.diffusion(noise, embedding=bert, features=cond)

        # --- 6) vocoder --------------------------------------------------------
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        wav = self.nets.decoder.mel2wav(mel).squeeze(1)          # [1,N]
        return wav


# ──────────────────────────────────────────────────────────────────────────
# 3.  Build model stack exactly like the repo
device = "cuda" if torch.cuda.is_available() else "cpu"
CFG, CKPT = "Models/LJSpeech/config.yml", "Models/LJSpeech/epoch_2nd_00100.pth"

cfg = yaml_to_munch(CFG)
if hasattr(cfg, "model_params"):
    cfg.update(cfg.model_params)

from Utils.PLBERT.util import load_plbert
from Utils.ASR.models   import ASRCNN
from Utils.JDC.model    import JDCNet

plbert = load_plbert("Utils/PLBERT")
plbert_ck = torch.load("Utils/PLBERT/step_1000000.t7", map_location="cpu")["net"]
plbert.load_state_dict({k.replace("module.encoder.","").replace("module.",""):v
                        for k,v in plbert_ck.items()}, strict=False)

asr = ASRCNN(80,256,178,6,512)
asr.load_state_dict(torch.load("Utils/ASR/epoch_00080.pth", map_location="cpu")["model"])

jdc = JDCNet(1)
jdc.load_state_dict(torch.load("Utils/JDC/bst.t7", map_location="cpu")["net"])

nets = build_model(cfg, asr, jdc, plbert)

ck = torch.load(CKPT, map_location="cpu")["net"]
for n,m in nets.items():
    part = {k[len(n)+1:]:v for k,v in ck.items() if k.startswith(n+".")}
    if part: m.load_state_dict(part, strict=False)

model = StyleTTS2Wrapper(nets, cfg).to(device).eval()

# ──────────────────────────────────────────────────────────────────────────
# 4.  Demo run
if __name__ == "__main__":
    os.makedirs("Demo", exist_ok=True)
    txt = "You never truly know how strong you are until being strong is your only choice."
    with torch.no_grad():
        wav = model(txt)
    torchaudio.save("Demo/output.wav", wav.cpu(), 24_000)
    print("✅  Saved waveform to Demo/output.wav")
