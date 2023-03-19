import os
import torch
import json
import joblib
import utils
from tqdm import tqdm
from multiprocessing.pool import Pool
from utils.hparams import hparams
# from modules.ProDiff.model.ProDiff_teacher import GaussianDiffusion
from .ProDiff_teacher import GaussianDiffusion
from usr.diff.net import DiffNet
from tasks.tts.fs2 import FastSpeech2Task
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder
from utils.pitch_utils import denorm_f0
from tasks.tts.fs2_utils import FastSpeechDataset
from utils.text_encoder import TokenTextEncoder
from .fs2 import FastSpeech2
# from .datasets_mask import LabelSpeechDataset
from .datasets_masksingle import LabelSpeechDatasetMaskSingleFairseq
from .datasets_masksingle_shard import LabelSpeechDatasetMaskSingleFairseqShard
from .datasets_masksingle_cv import LabelSpeechDatasetMaskSingleFairseqCV
from utils import audio
import librosa
from functools import partial
import random
import pathlib
from tasks.base_task import data_loader, BaseConcatDataset

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
}


class ProDiff_teacher_Task(FastSpeech2Task):
    def __init__(self):
        self.phone_list_file = hparams['phone_list_file']

        super(ProDiff_teacher_Task, self).__init__()
        random.seed(hparams['seed'])
        Dataset = eval( hparams.get('dataset_cls', "LabelSpeechDataset") )
        print('Use dataset:', Dataset)
        self.dataset_cls = partial(Dataset.build_dataset, self.phone_encoder)
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        # self.data_dir = hparams['binary_data_dir']
        # with open(f'{self.data_dir}/uttid2label.json', 'r') as f:
        #     self.uttid2label = json.load(f)
        self.spkembs = joblib.load(hparams['spkemb_path'])
        with open(hparams['spk_map'], 'r') as f:
            data = json.load(f)
            if type(data) == dict:
                self.seen_speakers = list(data.keys())
            elif type(data) == list:
                self.seen_speakers = data
            else:
                raise ValueError(f"{hparams['spk_map']} has type {type(data)}")
        # if not hasattr(self, 'generated_files'):

    def build_phone_encoder(self, data_dir):
        phone_list = []
        with open(self.phone_list_file, 'r') as f:
            for l in f:
                label = l.strip().split()[0]
                phone_list.append(label)
        assert len(phone_list) == 500, len(phone_list)
        return TokenTextEncoder(None, vocab_list=["<BOS>", "<EOS>", "<MASK>"]+phone_list, replace_oov=None)

    def build_model(self):
        self.build_tts_model()
        utils.num_params(self.model)
        return self.model

    def build_tts_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=hparams['audio_num_mel_bins'], denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'], time_scale=hparams['timescale'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )


    def run_model(self, model, sample, return_output=False, infer=False):
        # print(sample)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        f0 = None 
        uv = None
        energy = None
        # raise
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)
        # raise
        losses = {}
        self.add_mel_loss(output['mel_out'], target, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        energy = sample['energy']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']
        uv = sample['uv']

        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = utils.tensors_to_scalars(outputs)
        return outputs

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy() if f0 is not None else f0
        if is_mel:
            gt_wav = self.vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = self.vocoder.spec2wav(wav_out, f0=f0)
        self.logger.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)
        self.logger.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'], global_step=self.global_step)

    ############
    # inference
    ############
    def test_start(self):
        self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
        self.saving_results_futures = []
        self.results_id = 0
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'{hparams["gen_dir_name"]}_{self.trainer.global_step}')
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
        self.generated_files = set()
        for f in tqdm(pathlib.Path(self.gen_dir).glob('wavs/*.wav')):
            _, __, uttid, ___ = f.stem.split('__')
            self.generated_files.add(uttid)
        print('generated_files', len(self.generated_files))

    def test_step(self, sample, batch_idx):
        if hparams.get('random_spk', False) == True:
            true_spk_embed = sample.get('spk_embed')
            assert len(true_spk_embed) == 1, true_spk_emb.shape
            speaker = random.choice(self.seen_speakers)
            spk_embed = torch.from_numpy(self.spkembs[speaker]).to(true_spk_embed).unsqueeze(0)
        elif hparams.get('random_spk', False) == False:
            item_name = sample.get('item_name')
            speaker = item_name[0].split('-')[0] if item_name is not None else None
            spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        else:
            speaker = str(hparams['random_spk'])
            # print(f'use speaker {speaker}')
            true_spk_embed = sample.get('spk_embed')
            assert len(true_spk_embed) == 1, true_spk_emb.shape
            spk_embed = torch.from_numpy(self.spkembs[speaker]).to(true_spk_embed).unsqueeze(0)

        item_name = sample.get('item_name')[0]
        txt_tokens = sample['txt_tokens']
        # print('txttokens', txt_tokens)
        if item_name in self.generated_files:
            print(f'skip {self.gen_dir} / {item_name} as it has been generated...')
            return {}
        mel2ph, uv, f0 = None, None, None
        ref_mels = sample['mels']
        if hparams['use_gt_dur']:
            mel2ph = sample['mel2ph']
        if hparams['use_gt_f0']:
            f0 = sample['f0']
            uv = sample['uv']
        run_model = lambda: self.model(
            txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=uv, ref_mels=ref_mels, infer=True)
        if hparams['profile_infer']:
            mel2ph, uv, f0 = sample['mel2ph'], sample['uv'], sample['f0']
            with utils.Timer('fs', enable=True):
                outputs = run_model()
            if 'gen_wav_time' not in self.stats:
                self.stats['gen_wav_time'] = 0
            wav_time = float(outputs["mels_out"].shape[1]) * hparams['hop_size'] / hparams["audio_sample_rate"]
            self.stats['gen_wav_time'] += wav_time
            print(f'[Timer] wav total seconds: {self.stats["gen_wav_time"]}')
            from pytorch_memlab import LineProfiler
            with LineProfiler(self.model.forward) as prof:
                run_model()
            prof.print_stats()
        else:
            outputs = run_model()
            sample['speaker'] = [speaker]
            sample['outputs'] = self.model.out2mel(outputs['mel_out'])
            sample['mel2ph_pred'] = outputs['mel2ph']
            if hparams['use_pitch_embed']:
                sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
                if hparams['pitch_type'] == 'ph':
                    sample['f0'] = torch.gather(F.pad(sample['f0'], [1, 0]), 1, sample['mel2ph'])
                sample['f0_pred'] = outputs.get('f0_denorm')
            return self.after_infer(sample)
    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, alignment=None):
        # audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       # norm=hparams['out_wav_norm'])
        sr = hparams['audio_sample_rate']
        out_sr = 16000
        wav_out = librosa.resample(wav_out, orig_sr=sr, target_sr=out_sr)
        print('out_sr', out_sr, wav_out.shape)
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', out_sr,
                       norm=hparams['out_wav_norm'])

    def after_infer(self, predictions, sil_start_frame=0):
        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')
        mel_gt = prediction["mels"]
        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel_pred = prediction["outputs"]
        mel2ph_pred = prediction.get("mel2ph_pred")
        f0_gt = prediction.get("f0")
        f0_pred = prediction.get("f0_pred")
        speaker = prediction.get('speaker')
        str_phs = None
        encdec_attn = None
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'{self.results_id:06d}__{speaker}__{item_name}__%s'
        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            if 'encdec_attn' in prediction:
                os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred, encdec_attn]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph_gt]))
                # if hparams['save_f0']:
                #     import matplotlib.pyplot as plt
                #     f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                #     f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                #     fig = plt.figure()
                #     plt.plot(f0_pred_, label=r'$\hat{f_0}$')
                #     plt.plot(f0_gt_, label=r'$f_0$')
                #     plt.legend()
                #     plt.tight_layout()
                #     plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                #     plt.close(fig)
            print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()),
            'wav_fn_pred': base_fn % 'P',
            'wav_fn_gt': base_fn % 'G',
        }

    # @data_loader
    # def test_dataloader(self):
    #     print(1)
    #     test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
    #     print(2)
    #     self.test_dl = self.build_dataloader(
    #         test_dataset, False, self.max_valid_tokens,
    #         self.max_valid_sentences, batch_by_size=False)
    #     print(3)
    #     return self.test_dl
