#!/usr/bin/env python3
# Demo/infer_ljspeech.py  –  StyleTTS-2 inference (robust, no-pad)
import os, sys, yaml, torch, torchaudio
sys.path.insert(0, "/opt/apps/StyleTTS2")

from munch import Munch
from models import build_model

CTX_LEN   = 256          # fixed diffusion context length
HIDDEN_SZ = 1024         # transformer channel width

# ── yaml → Munch ───────────────────────────────────────────────────────
def yaml_to_munch(path):
    with open(path) as f: raw = yaml.safe_load(f)
    def rec(o):  return Munch({k:rec(v) for k,v in o.items()}) if isinstance(o,dict) \
                     else [rec(v) for v in o] if isinstance(o,list) else o
    return rec(raw)

# ── lightweight wrapper ────────────────────────────────────────────────
class StyleTTS2Wrapper(torch.nn.Module):
    def __init__(self, nets, cfg):
        super().__init__()
        self.nets = torch.nn.ModuleDict(nets)
        self.proj = torch.nn.Linear(1458, HIDDEN_SZ, bias=False)

    @staticmethod
    def _tok(text):
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?'-"
        table = {c:i+1 for i,c in enumerate(vocab)}
        ids = [table.get(c,0) for c in text.lower()][:CTX_LEN]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @torch.inference_mode()
    def forward(self, text: str) -> torch.Tensor:
        dev   = next(self.parameters()).device
        tok   = self._tok(text).to(dev)
        t     = tok.size(1)
        mask  = torch.zeros(1, t, dtype=torch.bool, device=dev)

        h_txt = self.nets.text_encoder(tok, torch.tensor([t],device=dev), mask)

        # style encoder
        if t >= 80:
            t16   = (t//16)*16 or 16
            sty_in= h_txt[:, :80, :t16].unsqueeze(1)
            style = self.nets.style_encoder(sty_in)
            h_txt = h_txt[:, :, :t16];  t_use = t16
        else:
            style = h_txt.new_zeros(1, self.nets.style_encoder.unshared.out_features)
            t_use = t

        # prosody
        eye          = torch.eye(t_use, device=dev).unsqueeze(0)
        dur, energy  = self.nets.predictor(h_txt, style,
                                           torch.tensor([t_use],device=dev),
                                           eye, mask[:, :t_use])
        energy = energy.transpose(1,2)
        pitch  = torch.zeros_like(energy)
        pros   = torch.cat((dur, pitch, energy), -1)

        style_time = style.unsqueeze(1).expand(-1, t_use, -1)        # ✅ add time-dim before expand
        cond  = torch.cat((pros, style_time), -1)
        cond  = self.proj(cond)                                      # [1,t,1024]

        # BERT
        bert = self.nets.bert(tok)[0]
        if bert.dim()==2: bert = bert.unsqueeze(0)
        bert = bert[:, :t_use, :]

        # pad / truncate to CTX_LEN
        if t_use < CTX_LEN:
            cond = torch.cat((cond, cond.new_zeros(1, CTX_LEN-t_use, HIDDEN_SZ)), 1)
            bert = torch.cat((bert, bert.new_zeros(1, CTX_LEN-t_use, 768)),        1)
        else:
            cond, bert = cond[:, :CTX_LEN], bert[:, :CTX_LEN]

        # diffusion – note noise shape [B,T,C]
        noise = torch.randn(1, CTX_LEN, HIDDEN_SZ, device=dev)       # ✅ last dim = channels
        mel   = self.nets.diffusion(noise, embedding=bert, features=cond)

        # handle possible channel layout
        if mel.dim()==4: mel = mel.squeeze(1)                        # [B,80,T]
        elif mel.size(-1)==80: mel = mel.permute(0,2,1)              # ✅ swap if [B,T,80]

        wav = self.nets.decoder.mel2wav(mel).squeeze(1)
        return wav

# ── build full stack just like repo ────────────────────────────────────
device  = "cuda" if torch.cuda.is_available() else "cpu"
CFG     = "Models/LJSpeech/config.yml"
CKPT    = "Models/LJSpeech/epoch_2nd_00100.pth"

cfg = yaml_to_munch(CFG)
if hasattr(cfg,"model_params"): cfg.update(cfg.model_params)

from Utils.PLBERT.util import load_plbert
from Utils.ASR.models  import ASRCNN
from Utils.JDC.model   import JDCNet

plbert = load_plbert("Utils/PLBERT")
plbert.load_state_dict({k.replace("module.encoder.","").replace("module.",""):v
                        for k,v in torch.load("Utils/PLBERT/step_1000000.t7",map_location="cpu")["net"].items()},
                       strict=False)

asr = ASRCNN(80,256,178,6,512)
asr.load_state_dict(torch.load("Utils/ASR/epoch_00080.pth",map_location="cpu")["model"])
jdc = JDCNet(1)
jdc.load_state_dict(torch.load("Utils/JDC/bst.t7",map_location="cpu")["net"])

nets = build_model(cfg, asr, jdc, plbert)
state= torch.load(CKPT,map_location="cpu")["net"]
for n,m in nets.items():
    sub = {k[len(n)+1:]:v for k,v in state.items() if k.startswith(n+".")}
    if sub: m.load_state_dict(sub, strict=False)

model = StyleTTS2Wrapper(nets,cfg).to(device).eval()

# ── quick test ─────────────────────────────────────────────────────────
if __name__=="__main__":
    os.makedirs("Demo",exist_ok=True)
    sentence = "You never truly know how strong you are until being strong is your only choice."
    with torch.no_grad():
        audio = model(sentence)
    torchaudio.save("Demo/output.wav", audio.cpu(), 24000)
    print("✅  Saved to Demo/output.wav")
