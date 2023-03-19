import os
import json
import joblib
import matplotlib
import random
matplotlib.use('Agg')

import glob
import importlib
from utils.cwt import get_lf0_cwt
import torch.optim
import torch.utils.data
from utils.indexed_datasets import IndexedDataset
from utils.pitch_utils import norm_interp_f0
import numpy as np
from tasks.base_task import BaseDataset
import torch
import torch.optim
import torch.utils.data
import utils
import torch.distributions
from utils.hparams import hparams
from fairseq.data.data_utils import compute_mask_indices
from utils.text_encoder import TokenTextEncoder
from tqdm import tqdm
from pathlib import Path

class LabelSpeechDatasetMaskSingleFairseqCV(BaseDataset):
    def __init__(self, phone_encoder, prefix, shuffle=False):
        super().__init__(shuffle)
        random.seed(hparams['seed'])
        # split = hparams[f'{prefix}_set_name']
        self.prefix = prefix
        self.is_infer = hparams['infer']
        self.feat_nshard = hparams[f'feat_{prefix}_nshard']
        self.label_dir = hparams['label_dir']
        self.feat_dir = hparams['feat_dir']
        self.sr = hparams['audio_sample_rate'] # number of samples per second
        # self.aux_context_window = hparams['aux_context_window']
        # self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.spkemb_path = hparams['spkemb_path']
        
        self.uttid2spk_path = hparams['uttid2spk_path']
        print(f'use {self.uttid2spk_path}')
        self.hop_size = hparams['hop_size']
        self.feat_rate = self.sr / self.hop_size # number of features per second
        self.label_resample_rate = np.prod(hparams['us_stride']) / np.prod(hparams['ds_stride'])

        self.labels = []
        with open(f'{self.label_dir}/{self.prefix}.km', 'r') as f:
            for l in f:
                self.labels.append(l.strip())

        self.feats = []
        for rank in range(self.feat_nshard):
            feat_path = f'{self.feat_dir}/{self.prefix}_{rank}_{self.feat_nshard}.npy'
            feats = np.load(feat_path, mmap_mode="r")
            self.feats.append(feats)

        self.feat_lengs, self.feat_ranks, self.feat_offsets = [], [], []
        for rank in range(self.feat_nshard):
            leng_path = f'{self.feat_dir}/{self.prefix}_{rank}_{self.feat_nshard}.len'
            with open(leng_path, 'r') as f:
                lengs = [int(line.rstrip()) for line in f]
                offests = [0] + np.cumsum(lengs[:-1]).tolist()
                ranks = [rank] * len(lengs)
                self.feat_lengs += lengs
                self.feat_offsets += offests
                self.feat_ranks += ranks

        manifest_dir = hparams['manifest_dir']
        manifest = f'{manifest_dir}/{self.prefix}.tsv'
        self.wav_paths, self.wav_lengs = [], []
        with open(manifest, 'r') as f:
            root_dir = f.readline().strip()
            for l in tqdm(f):
                rel_path, length = l.strip('\n').split('\t')
                self.wav_paths.append(os.path.join(root_dir, rel_path))
                self.wav_lengs.append(int(length))

        # filter out short utterances
        # filtered_indices = []
        # for ix, feat_leng in enumerate(self.feat_lengs):
        #     if feat_leng - 2 * self.aux_context_window > self.batch_max_frames:
        #         filtered_indices.append(ix)
        # self.filtered_indices = filtered_indices
        filtered_indices = list(range(len(self.feat_lengs)))
        print(f'{len(filtered_indices)}/{len(self.feat_lengs)} samples remaining after filtering')

        self.labels = np.array(self.labels)[filtered_indices]
        self.feat_lengs, self.feat_ranks, self.feat_offsets = (
            np.array(self.feat_lengs)[filtered_indices], 
            np.array(self.feat_ranks)[filtered_indices], 
            np.array(self.feat_offsets)[filtered_indices]
        )
        self.wav_paths, self.wav_lengs = (
            np.array(self.wav_paths)[filtered_indices], 
            np.array(self.wav_lengs)[filtered_indices]
        )

        assert len(self.wav_paths) == len(self.feat_lengs), \
            f'len(self.wav_paths) {len(self.wav_paths)} == len(self.feat_lengs) {len(self.feat_lengs)}'
        for wav_path, wav_leng, feat_leng in zip(self.wav_paths, self.wav_lengs, self.feat_lengs):
            predicted_feat_leng = int(wav_leng / self.sr * self.feat_rate)
            assert np.abs(predicted_feat_leng - feat_leng) <= 10, \
                f'{wav_path} has sample length {wav_leng} and feat length {feat_leng}'

        self.sizes = self.feat_lengs
        
        self.spkembs = joblib.load(self.spkemb_path)
        with open(self.uttid2spk_path, 'r') as f:
            self.uttid2spk = json.load(f)

        self.prefix = prefix
        self.hparams = hparams

        self.mask_prob = hparams['mask_prob']
        self.mask_selection = 'static'
        self.mask_other = 0
        self.no_mask_overlap = False
        self.mask_min_space = 1
        self.phone_encoder = phone_encoder
        self.mask_idx = self.phone_encoder.encode('<MASK>')[0]
        self.mask_partial_threshold = None
        ps = hparams.get('mask_partial_probs') #  [ do not mask, mask all, partially mask ]
        if ps is not None:
            self.mask_partial_threshold = [ps[0], ps[0]+ps[1]]
            print('mask_partial_threshold', self.mask_partial_threshold)

    @classmethod
    def build_dataset(cls, phone_encoder, prefix, shuffle=False):
        return cls(phone_encoder, prefix, shuffle)

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]
    
    def mask_phone(self, phone):
        B = 1
        T = len(phone)
        hparams = self.hparams
        mask_indices = torch.zeros(T).bool()
        rd = random.uniform(0,1)

        if rd < self.mask_partial_threshold[0]:
            # do not mask
            mask_indices[:] = False
        elif rd < self.mask_partial_threshold[1]:
            # mask all
            mask_indices[:] = True
        else:
            span_len = int(len(phone) * self.mask_prob)
            startid = np.random.choice(list(range(0,len(phone)-span_len)), )
            endid = startid + span_len
            mask_indices = torch.zeros(T).bool() # all false
            mask_indices[startid:endid] = True
            # partially mask use compute_mask_indices results
                
        phone[mask_indices] = self.mask_idx
        return phone

    
    def __getitem__(self, index):
        hparams = self.hparams
        wav_path = self.wav_paths[index]
        item_name = Path(wav_path).stem 
        spkid = self.uttid2spk[item_name]

        rank = self.feat_ranks[index]
        offset, leng = self.feat_offsets[index], self.feat_lengs[index]
        assert offset+leng <= len(self.feats[rank]), f'{offset}+{leng} <= {len(self.feats[rank])}'
        feat = self.feats[rank][offset:offset+leng].copy()
        if len(feat.shape) == 3:
            feat = feat.squeeze(-1)

        label = self.labels[index]
        phone = torch.LongTensor(self.phone_encoder.encode(f'{label}'))
        phone = self.mask_phone(phone)

        spec = mel = torch.FloatTensor(feat)
        # assert len(mel) == len(phone), f'{item_name} {len(mel)} == {len(phone)}'
        mel2ph = np.linspace(0, len(mel), len(mel), endpoint=False).astype(int)+1
        mel2ph = torch.LongTensor(mel2ph)
        assert len(mel2ph) == len(mel), f'{item_name}: {len(mel2ph)} == {len(mel)}'

        label_leng = int(len(phone) * self.label_resample_rate)
        assert np.abs(label_leng - len(mel2ph)) <= 5, f'abs({label_leng} - {len(mel2ph)}) <= 5'
        T = min(label_leng, len(mel2ph))
        if label_leng < len(mel2ph):
            mel2ph, spec = mel2ph[:T], spec[:T]
        elif label_leng > len(mel2ph):
            print(mel2ph.shape, phone.shape, label_leng)
            raise
        spk_embed = self.spkembs[spkid]

        sample = {
            "id": index,
            "item_name": item_name,
            "text": label,
            "txt_token": phone,
            "mel": spec,
            "mel2ph": mel2ph,
            "pitch": None,
            "energy": None,
            "f0": None,
            "uv": None,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.FloatTensor(spk_embed)
        if self.hparams['use_spk_id']:
            sample["spk_id"] = spkid
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples], 0)

        mel2ph = utils.collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = utils.collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': None,
            'pitch': None,
            'f0': None,
            'uv': None,
        }

        if self.hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if self.hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        return batch


