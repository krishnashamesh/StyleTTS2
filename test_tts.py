import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import os, pathlib

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize

import soundfile as sf

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

out_dir = "embedding"
os.makedirs(out_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Env hooks
#  - REF_WAV: single path (backward compatible)
#  - REF_WAVS: comma/whitespace separated list of paths
#  - REF_LIST: path to a txt file with one path per line
REF_WAV  = os.environ.get("REF_WAV", "").strip()
REF_WAVS = os.environ.get("REF_WAVS", "/opt/apps/StyleTTS2/Data/VCTK/wav_norm/p225/p225_023.wav, /opt/apps/StyleTTS2/Data/VCTK/wav_norm/p226/p226_023.wav, /opt/apps/StyleTTS2/Data/VCTK/wav_norm/p248/p248_023.wav, /opt/apps/StyleTTS2/Data/VCTK/wav_norm/p251/p251_023.wav, /opt/apps/StyleTTS2/Data/VCTK/wav_norm/p376/p376_023.wav").strip()
REF_LIST = os.environ.get("REF_LIST", "").strip()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# ===== Helpers to gather reference paths and pretty names =====
def _split_paths(s: str):
    """Split on commas or whitespace; keep only non-empty strings."""
    if not s:
        return []
    parts = []
    for chunk in s.split(','):
        parts.extend(chunk.strip().split())
    return [p for p in map(str.strip, parts) if p]

def _read_list_file(p: str):
    if not p:
        return []
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"[ms] WARN: failed to read REF_LIST '{p}': {e}")
        return []

def gather_ref_paths():
    paths = []
    paths.extend(_split_paths(REF_WAVS))
    paths.extend(_read_list_file(REF_LIST))
    if REF_WAV:
        paths.append(REF_WAV)
    # de-dup while preserving order
    seen = set(); out = []
    for p in paths:
        if p not in seen: seen.add(p); out.append(p)
    return out

# ===== Multi-speaker reference → features (Style ⊕ Predictor) =====
@torch.no_grad()
def compute_ref_features_from_wav(ref_wav_path: str, sr: int = 24000):
    """
    Build the context features expected by the multi-speaker diffusion UNet:
      ref_feat = concat(style_encoder(ref_mel), predictor_encoder(ref_mel))  # [1, 256]
    """
    wave, in_sr = librosa.load(ref_wav_path, sr=sr)
    # light trim; keep small leading context
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if in_sr != sr:
        audio = librosa.resample(audio, in_sr, sr)
    mel = preprocess(audio).to(device)                 # [1, 80, T]
    # encoders expect [B, 1, 80, T]
    ref_ss = model.style_encoder(mel.unsqueeze(1))     # [1, 128]
    ref_sp = model.predictor_encoder(mel.unsqueeze(1)) # [1, 128]
    return torch.cat([ref_ss, ref_sp], dim=1)          # [1, 256]

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)

    return reference_embeddings

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

config = yaml.safe_load(open("Models/VCTK/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

# Detect multispeaker from config
MULTISPEAKER = bool(config.get('model_params', {}).get('multispeaker', False))
if MULTISPEAKER:
    print("[ms] Multispeaker mode: expecting at least one reference audio (REF_WAV / REF_WAVS / REF_LIST).")
else:
    print("[ms] Single-speaker mode (no reference features required).")

params_whole = torch.load("Models/VCTK/epoch_2nd_00004.pth")
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def inference(text, noise, diffusion_steps=5, embedding_scale=1, ref_feat: torch.Tensor | None = None):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        # If multi-speaker, pass context features from reference audio
        if MULTISPEAKER and (ref_feat is not None):
            s_pred = sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                features=ref_feat,
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale
            ).squeeze(0)
        else:
            s_pred = sampler(noise, embedding=bert_dur[0].unsqueeze(0),
                             num_steps=diffusion_steps, embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()

def LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1, ref_feat: torch.Tensor | None = None):
  text = text.strip()
  text = text.replace('"', '')
  ps = global_phonemizer.phonemize([text])
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)

  tokens = textclenaer(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
      text_mask = length_to_mask(input_lengths).to(tokens.device)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      if MULTISPEAKER and (ref_feat is not None):
          s_pred = sampler(
              noise,
              embedding=bert_dur[0].unsqueeze(0),
              features=ref_feat,
              num_steps=diffusion_steps,
              embedding_scale=embedding_scale
          ).squeeze(0)
      else:
          s_pred = sampler(noise,
                           embedding=bert_dur[0].unsqueeze(0),
                           num_steps=diffusion_steps, embedding_scale=embedding_scale).squeeze(0)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = alpha * s_prev + (1 - alpha) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)
      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)

      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
      out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))

  return out.squeeze().cpu().numpy(), s_pred


# synthesize a text
text = "I was not asking for magic. Just honesty. Just a moment that felt real. "


# === Run inference ===
if not MULTISPEAKER:
    # single-speaker path: do exactly one synthesis as before
    start = time.time()
    noise = torch.randn(1,1,256).to(device)
    wav = inference(text, noise, diffusion_steps=20, embedding_scale=1, ref_feat=None)
    time_taken = (time.time() - start)
    rtf = time_taken / (len(wav) / 24000)
    print(f"RTF = {rtf:5f}")
    print(f"Time Taken (single-speaker, 20 steps, es=1): {time_taken:5f}")
    sf.write(os.path.join(out_dir, "output_20steps_es1.wav"), wav, 24000)
else:
    ref_paths = gather_ref_paths()
    if not ref_paths:
        raise RuntimeError("[ms] Multispeaker: no references found. "
                           "Set REF_WAV, or REF_WAVS (comma/space list), or REF_LIST (file with one path per line).")
    print(f"[ms] Found {len(ref_paths)} reference file(s).")
    for idx, ref_path in enumerate(ref_paths, 1):
        if not os.path.exists(ref_path):
            print(f"[ms] WARN: reference does not exist, skipping: {ref_path}")
            continue
        try:
            ref_feat = compute_ref_features_from_wav(ref_path).to(device)  # [1, 256]
        except Exception as e:
            print(f"[ms] WARN: failed to build ref features for '{ref_path}': {e}")
            continue
        # nice stub for filename: <parent-name>_<stem> (e.g., p225_p225_003)
        p = pathlib.Path(ref_path)
        parent = p.parent.name or "spk"
        stub = f"{parent}_{p.stem}"
        # synth
        start = time.time()
        noise = torch.randn(1,1,256).to(device)
        wav = inference(text, noise, diffusion_steps=20, embedding_scale=1, ref_feat=ref_feat)
        time_taken = (time.time() - start)
        rtf = time_taken / (len(wav) / 24000)
        print(f"[ms] [{idx}/{len(ref_paths)}] {stub}: RTF={rtf:5f}  time={time_taken:5f}")
        out_name = os.path.join(out_dir, f"{stub}_20steps_es1.wav")
        sf.write(out_name, wav, 24000)
    print("[ms] Done.")