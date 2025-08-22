#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from collections import OrderedDict

from collections import defaultdict

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 mel_cache_dir=None,
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr
        self.mel_cache_dir = mel_cache_dir

        self.df = pd.DataFrame(self.data_list)

        # Pre-group items by speaker for fast random ref sampling (avoids per-item pandas.sample)
        self.by_spk = defaultdict(list)
        for p, t, s in self.data_list:
            try:
                sid = int(s)
            except Exception:
                try: sid = int(str(s).strip())
                except Exception: sid = 0
            self.by_spk[sid].append((p, t, sid))

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192

        # Per-process LRU cache for decoded waves (uses host RAM)
        self._cache = OrderedDict()
        # Tune via env var; this is a count of wave entries (not MB)
        # With N workers, total cached entries ≈ N * DATASET_CACHE_CAP
        self._cache_cap = int(os.environ.get("DATASET_CACHE_CAP", "4096"))
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path

    def _get_wave(self, wave_path_full):
        """LRU-cached wave loader (mono @ 24kHz)."""
        if wave_path_full in self._cache:
            x = self._cache.pop(wave_path_full)   # move to MRU
            self._cache[wave_path_full] = x
            return x
        wave, sr = sf.read(wave_path_full)
        if wave.ndim == 2 and wave.shape[-1] == 2:
            wave = wave[:, 0]
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        # pad to allow crops near edges (matches original behavior)
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        # LRU insert
        try:
            self._cache[wave_path_full] = wave
            if len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)   # evict LRU
        except Exception:
            pass
        return wave

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        # try cache first; fall back to on-the-fly compute
        mel_tensor = self._load_or_make_mel(path, wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get reference sample
        # Fast ref sample from pre-built speaker index
        spk = int(speaker_id) if not isinstance(speaker_id, int) else speaker_id
        pool = self.by_spk.get(spk) or [(path, data[1], spk)]
        ref_p, ref_t, _ = random.choice(pool)
        ref_mel_tensor, ref_label = self._load_data((ref_p, ref_t, spk))
        
        # get OOD text
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)
        
        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave_path_full = osp.join(self.root_path, wave_path)
        wave = self._get_wave(wave_path_full)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):

        wave_path, text, speaker_id = data
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = self._load_or_make_mel(wave_path, wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id
    
    def _load_or_make_mel(self, rel_path, wave):
        """
        If mel_cache_dir is set, load <mel_cache_dir>/<rel_path>.pt (or .npy).
        Otherwise compute inline using preprocess(wave).
        """
        if self.mel_cache_dir:
            try:
                tgt = osp.splitext(osp.join(self.mel_cache_dir, rel_path))[0]
                pt_path = tgt + ".pt"
                if osp.exists(pt_path):
                    return torch.load(pt_path, map_location="cpu").float()
                npy_path = tgt + ".npy"
                if osp.exists(npy_path):
                    arr = np.load(npy_path)
                    return torch.from_numpy(arr).float()
            except Exception as e:
                # fall through to compute
                pass
        return preprocess(wave).float()


class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels


#Add support for persistent workers
def build_dataloader(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     prefetch_factor=4,
                     persistent_workers=None,
                     timeout=0,
                     pin_memory_device=None,
                     collate_config={},
                     dataset_config={}):
    
    def _worker_init_fn(_):
        try:
            import torch, os
            torch.set_num_threads(1)
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
        except Exception:
            pass    

    dataset = FilePathDataset(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)

    use_pin = (device != 'cpu')
    if pin_memory_device is None and use_pin:
        pin_memory_device = 'cuda'
    if persistent_workers is None:
        persistent_workers = (num_workers > 0 and not validation)

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=use_pin,
                             pin_memory_device=pin_memory_device if use_pin else '',
                             persistent_workers=persistent_workers,
                             prefetch_factor=prefetch_factor if num_workers > 0 else 2,
                             worker_init_fn=_worker_init_fn,
                             timeout=timeout,)

    return data_loader

