# DiffS4L
This repo contains code and materials for NeurIPS2023 submission. The DiffS4L requires [Fairseq](https://github.com/facebookresearch/fairseq) and [ProDiff](https://github.com/Rongjiehuang/ProDiff) and optionally [WaveNet](https://github.com/r9y9/wavenet_vocoder) for baseline.
## 1. Dependencies
### 1.1 Install Fairseq
```
wget https://github.com/facebookresearch/fairseq/archive/refs/tags/v0.12.2.zip
unzip v0.12.2.zip
cd fairseq-0.12.2
pip install --editable ./
```
### 1.2 Download ProDiff
```
git clone https://github.com/Rongjiehuang/ProDiff.git
```
### 1.3 Install SCTK for CER/WER evaluation
Follow the instruction in [SCTK](https://github.com/usnistgov/SCTK) repo to install SCTK

## 2. DiffS4L Pipeline
### 2.1 Pretrain Wav2vec on 100 hour LibriSpeech
Download Librispeech data from [openSLR](https://www.openslr.org/12).
```
wget https://us.openslr.org/resources/12/train-clean-100.tar.gz
wget https://us.openslr.org/resources/12/dev-clean.tar.gz
wget https://us.openslr.org/resources/12/test-clean.tar.gz
tar -xf train-clean-100.tar.gz
tar -xf dev-clean.tar.gz
tar -xf test-clean.tar.gz
```
Create manifest files ``train.tsv``, ``dev.tsv`` and ``test.tsv`` for ``train-clean-100``, ``dev-clean`` and ``test-clean`` splits respectively using the script [wav2vec_manifest.py](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py) provided by Fairseq.
```
mkdir -p manifest/LibriSpeech100
cd manifest/LibriSpeech100
python examples/wav2vec/wav2vec_manifest.py LibriSpeech/dev-clean --dest manifest/LibriSpeech100 --ext flac --valid-percent 0
mv train.tsv dev.tsv
python examples/wav2vec/wav2vec_manifest.py LibriSpeech/test-clean --dest manifest/LibriSpeech100 --ext flac --valid-percent 0
mv train.tsv test.tsv
python examples/wav2vec/wav2vec_manifest.py LibriSpeech/train-clean-100 --dest manifest/LibriSpeech100 --ext flac --valid-percent 0
```
Pretrain a Wav2vec model using LibriSpeech100.
```
cd fairseq-0.12.2/examples/wav2vec
seed=2023
WORK_DIR=$(pwd)
DATASET=LibriSpeech100
MANIFEST_DIR=manifest/${DATASET}
save_dir=outputs/wav2vec-en-${DATASET}-base-s${seed}
fairseq-hydra-train \
    distributed_training.distributed_port=12345 \
    distributed_training.distributed_world_size=32 \
    task.data=${MANIFEST_DIR} \
    checkpoint.save_dir=${save_dir} hydra.run.dir=${save_dir} \
    common.fp16=True common.seed=${seed} \
    +optimization.update_freq='[2]' \
    --config-dir ${WORK_DIR}/config/pretraining \
    --config-name wav2vec2_base_librispeech
```
### 2.2 Extract discrete speech representation from Wav2vec features.
Follow the instruction in [HuBERT repo](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert/simple_kmeans) to obtain discrete speech representations using the ``wav2vec-en-LibriSpeech100-base-s2023`` model for ``train-clean-100``, ``dev-clean`` and ``test-clean`` and name them ``train.km``, ``dev.km`` and ``test.km``. I store the three km files in ``manifest/LibriSpeech100/wav2vec-ls100``.
```
mkdir -p manifest/LibriSpeech100/wav2vec-ls100
mv train.km dev.km test.km manifest/LibriSpeech100/wav2vec-ls100
```

### 2.3 Extract melspectrogram for LibriSpeech100.
Extract melspectrogram of LibriSpeech100. The ``ckpt_path`` here is not used but just a place holder.
```
ckpt_path="wav2vec-en-LibriSpeech100-base-s2023/checkpoint_best.pt"
model_name="mel"
layer=0
sample_rate=16000

FEATURE_SCRIPT="dump_mel_feature.py"
km_dataset="LibriSpeech100"

tsv_dir="$(pwd)/manifest/${km_dataset}"
feat_dir="$(pwd)/feats/${model_name}_$(basename ${km_dataset})_l${layer}"

mkdir -p ${feat_dir}

pids=()
splits=(train dev test)
for split in ${splits[@]}; do
    nshard=8
    ranks=$(seq 0 $((nshard - 1)))
    for rank in ${ranks[@]}; do
        (
            echo "Parallel ${rank}"
            python -W ignore ${FEATURE_SCRIPT} \
                "${tsv_dir}" "${split}" \
                "${ckpt_path}" "${layer}" "${nshard}" "${rank}" "${feat_dir}" --sample_rate $sample_rate
        ) &
        pids+=($!)
    done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
```

### 2.4 Train DDPM using LibriSpeech100 and its km Labels
Move the ProDiff/egs to the ProDiff cloned from GitHub in Sec 1.2. ``librivox_en_mask0.yaml`` is the DDPM for SS/DS speech and the ``librivox_en_mask0.8.yaml`` is the DDPM for NC speech. Change ``label_dir`` to the folder storing the km labels, i.e. ``label_dir=manifest/LibriSpeech100/wav2vec-ls100``. ``spkemb_path``, ``spk_map`` and ``phone_list_file`` can be found in ``manifest/LibriSpeech100``
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/run.py --config egs/datasets/audio/librivox/prodiff/librivox_en_mask0.8.yaml --exp_name librivox_en_mask0.8 --reset
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tasks/run.py --config egs/datasets/audio/librivox/prodiff/librivox_en_mask0.yaml --exp_name librivox_en_mask0 --reset
```
### 2.5 Generate SS/DS and NC speech using DDPM
Generate SS/DS and NC synthetic audio datasets that are both 10 times the size of original LibriSpeech100 audio using different seeds. ``librivox_en_mask0.yaml`` is the DDPM for SS/DS speech and the ``librivox_en_mask0.8.yaml`` is the DDPM for NC speech. Our HiFiGAN trained on LibriSpeech100 can be downloaded [[here]](https://drive.google.com/drive/folders/17UNrm6hZiiMWh8WLgJ7ui9XUZnjF2mNA?usp=share_link).
```
pids=()
seeds=(1 2 3 4 5 6 7 8 9 10)
for si in ${!seeds[@]}; do
    seed=${seeds[$si]}
    rand_speaker=True
    use_speaker=seen
    (
    vocoder='hifigan'
    vocoder_ckpt='mls_librispeech100'
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
        --config egs/datasets/audio/librivox/prodiff/librivox_en_mask0.yaml \
        --exp_name librivox_en_mask0 --infer \
        --hparams="save_gt=False,test_set_name=train,num_test_samples=0,random_spk=True,seed=${seed},gen_dir_name='generate/${use_speaker}${seed}',mask_partial_probs=[1 0 0],vocoder_ckpt=${vocoder_ckpt},vocoder=${vocoder}"
        
        PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python tasks/run.py \
        --config egs/datasets/audio/librivox/prodiff/librivox_en_mask0.8.yaml \
        --exp_name librivox_en_mask0.8 --infer \
        --hparams="save_gt=False,test_set_name=train,num_test_samples=0,random_spk=${rand_speaker},seed=${seed},gen_dir_name='generate/${use_speaker}${seed}',mask_partial_probs=[0 0 1],vocoder_ckpt=${vocoder_ckpt},vocoder=${vocoder}"
    ) &
    pids+=($!)
done

i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
```
### 2.6 Train DiffS4L
Sample from the audios generated by ``librivox_en_mask0`` for SS/DS speech and from the audios generated by ``librivox_en_mask0.8`` for NC speech. The sampled audio combined with the original ``train-clean-100`` LibriSpeech dataset makes up the different data composition, e.g., ``100+860+0`` and ``100+430+430``. The ``dev.tsv`` and ``test.tsv`` are the same as the ones in LibriSpeech100. 
```
cd fairseq-0.12.2/examples/wav2vec
seed=2023
WORK_DIR=$(pwd)
DATASET="100+430+430"
MANIFEST_DIR=manifest/${DATASET}
save_dir=outputs/wav2vec-en-${DATASET}-base-s${seed}
fairseq-hydra-train \
    distributed_training.distributed_port=12345 \
    distributed_training.distributed_world_size=32 \
    task.data=${MANIFEST_DIR} \
    checkpoint.save_dir=${save_dir} hydra.run.dir=${save_dir} \
    common.fp16=True common.seed=${seed} \
    +optimization.update_freq='[2]' \
    --config-dir ${WORK_DIR}/config/pretraining \
    --config-name wav2vec2_base_librispeech
```
### 2.7 Evaluate Wav2vec-100R and Wav2vec-DiffS4L on ASR
Use [libri_labels.py](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/libri_labels.py) to obtain transcripts for ``dev-clean`` and ``test-clean`` splits. The label dictionary can be found [[here]](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt).
```
python libri_labels.py dev.tsv --output-dir . --output-name dev
python libri_labels.py test.tsv --output-dir . --output-name test
mv dev.ltr test.ltr manifest/LibriSpeech100
```
Evaluate on ``dev-clean`` and ``test-clean`` splits using CER/WER. The ``lexicon_ltr.lst`` can be downloaded [[here]](https://dl.fbaipublicfiles.com/textless_nlp/gslm/eval_data/lexicon_ltr.lst). The ``4-gram.bin`` can be downloaded [[here]](https://drive.google.com/file/d/1eLWJQwnk5hYVbdeHuxcuh7zEkSOmlpZc/view?usp=share_link).
```
ckpt=/path/to/wav2vec/checkpoint
manifest=manifest/LibriSpeech100
KENLM=4-gram.bin
LEXICON=lexicon_ltr.lst
SCTK=SCTK/bin

splits=(dev test)
FAIRSEQ=fairseq-0.12.2
for split in ${splits[@]}; do
    subset=$split
    result_dir=$(dirname $ckpt)/results_raw
    mkdir -p $result_dir
    echo --------------------- Save $subset raw results in $result_dir -----------------------
    python $FAIRSEQ/examples/speech_recognition/infer.py \
        $manifest --task audio_finetuning \
        --nbest 1 --path $ckpt --gen-subset $subset --results-path ${result_dir} --w2l-decoder viterbi \
        --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter --beam 1 \
        2> $result_dir/log-${subset}.txt

    $SCTK/sclite -r ${result_dir}/ref.units-checkpoint_best.pt-${subset}.txt trn -h ${result_dir}/hypo.units-checkpoint_best.pt-${subset}.txt trn -i wsj | tee ${result_dir}/uer-${subset}.txt
    $SCTK/sclite -r ${result_dir}/ref.word-checkpoint_best.pt-${subset}.txt trn -h ${result_dir}/hypo.word-checkpoint_best.pt-${subset}.txt trn -i wsj | tee ${result_dir}/wer-${subset}.txt

    echo --------------------------------------------------------------------------


    result_dir=$(dirname $ckpt)/results_kenlm
    mkdir -p $result_dir
    echo --------------------- Save $subset kenlm results in $result_dir -----------------------
    python $FAIRSEQ/examples/speech_recognition/infer.py \
        $manifest --task audio_finetuning \
        --nbest 1 --path $ckpt --gen-subset $subset --results-path ${result_dir} --w2l-decoder kenlm --lexicon ${LEXICON} \
        --lm-model $KENLM --lm-weight 1 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
        --post-process letter 2> $result_dir/log-${subset}.txt

    $SCTK/sclite -r ${result_dir}/ref.units-checkpoint_best.pt-${subset}.txt trn -h ${result_dir}/hypo.units-checkpoint_best.pt-${subset}.txt trn -i wsj | tee ${result_dir}/uer-${subset}.txt
    $SCTK/sclite -r ${result_dir}/ref.word-checkpoint_best.pt-${subset}.txt trn -h ${result_dir}/hypo.word-checkpoint_best.pt-${subset}.txt trn -i wsj | tee ${result_dir}/wer-${subset}.txt
    echo --------------------------------------------------------------------------


done
```
### 2.8 Evaluate Wav2vec-100R and Wav2vec-DiffS4L on SUPERB benchmark
Follow the instructions in [SUPERB](https://github.com/s3prl/s3prl) to evaluate the pretrained SSL models.
