#!/bin/bash
PYTHON_VIRTUAL_ENVIRONMENT=wavenet-r9y9
CONDA_ROOT=???

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
VOC_DIR=$script_dir/../../

# Directory that contains all wav files
# **CHANGE** this to your database path
db_root=metadata/LibriSpeech100
spk="lsp"
dumpdir=dump

# train/dev/eval split
dev_size=30
eval_size=30
# Maximum size of train/dev/eval data (in hours).
# set small value (e.g. 0.2) for testing
limit=1000000

# waveform global gain normalization scale
global_gain_scale=0.55

stage=0
stop_stage=0

# Hyper parameters (.json)
# **CHANGE** here to your own hparams
hparams=conf/mulaw256_wavenet.json
# hparams=conf/mulaw256_wavenet_demo.json

# Batch size at inference time.
inference_batch_size=10
# Leave empty to use latest checkpoint
eval_checkpoint=
# Max number of utts. for evaluation( for debugging)
eval_max_num_utt=1000000

# exp tag
tag="lsp100w2vnv" # tag for managing experiments.

total_generation_hour=16
seed=0
device=cuda
duration=6
temperature=1
teacher_dur=0.3
threshold=0.1
rank=0
nshard=1
filter_lambda="lambda x: x < 13"
max_token=320000
km_path=
generation_tag=
n_batch=10
nprocess=8
audio_dir=
use_speakers=same

. $VOC_DIR/utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_no_dev"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)

# exp name
if [ -z ${tag} ]; then
    expname=${spk}_${train_set}_$(basename ${hparams%.*})
else
    expname=${spk}_${train_set}_${tag}
fi
expdir=exp/$expname

feat_typ="logmelspectrogram"

# Output directories
data_root=data/$spk                        # train/dev/eval splitted data
dump_org_dir=$dumpdir/$spk/$feat_typ/org   # extracted features (pair of <wave, feats>)
# dump_norm_dir=$dumpdir/$spk/$feat_typ/norm # extracted features (pair of <wave, feats>)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature Generation"
	manifest_dir="$(pwd)/metadata/LibriSpeech100"
    for s in ${datasets[@]};
    do
      python $VOC_DIR/preprocess.py librivox ${manifest_dir}/${s}.tsv ${dump_org_dir}/$s \
        --hparams="global_gain_scale=${global_gain_scale}" --preset=$hparams --num_workers=64
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: WaveNet training"
    python $VOC_DIR/train_noval_v2.py --dump-root $dump_org_dir --preset $hparams \
      --checkpoint-dir=$expdir \
      --log-event-path=tensorboard/${expname} \
      --checkpoint=$(pwd)/exp/${spk}_train_no_dev_${tag}/checkpoint_latest.pth
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Synthesis waveform from WaveNet"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    for s in $eval_set $dev_set;
    do
      dst_dir=$expdir/generated/$name/${s}_3s_gpubsz30
      python $VOC_DIR/evaluate.py $dump_org_dir/$s $eval_checkpoint $dst_dir \
        --preset $hparams --hparams="batch_size=$inference_batch_size" \
        --num-utterances=$eval_max_num_utt
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Synthesis lots of waveforms from WaveNet (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_dur${duration}_bsz${inference_batch_size}_${seed}_tdur0_ttt${temperature}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size" --duration=${duration} \
      --total-hour=${total_generation_hour} --seed=${seed} --temperature=${temperature} --device=${device} \
      $eval_checkpoint $dst_dir
fi

if [ ${stage} -le 41 ] && [ ${stop_stage} -ge 41 ]; then
    echo "stage 41: Synthesis lots of waveforms from WaveNet (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_stage41_dur${duration}_bsz${inference_batch_size}_${seed}_tdur0_thd${threshold}_tttt${temperature}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size" --duration=${duration} \
      --total-hour=${total_generation_hour} --seed=${seed} --temperature=${temperature} --device=${device} --threshold=${threshold} --nbatch=1 \
      $eval_checkpoint $dst_dir
fi

if [ ${stage} -le 42 ] && [ ${stop_stage} -ge 42 ]; then
    echo "stage 42: Synthesize one batch waveforms from Hubert features (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_stage42_dur${duration}_bsz${inference_batch_size}_${seed}_t${temperature}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v3.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size,num_workers=2" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=1 \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank}
fi

if [ ${stage} -le 43 ] && [ ${stop_stage} -ge 43 ]; then
    echo "stage 43: Synthesize batches waveforms from Hubert features (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/wav_gen/${s}_stage43_dur${duration}_bsz${inference_batch_size}_r${rank}_t${temperature}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v3.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size,num_workers=6" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank}
fi

if [ ${stage} -le 431 ] && [ ${stop_stage} -ge 431 ]; then
    echo "stage 431: Synthesize batches waveforms from Hubert features (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    ignore_list=$(pwd)/wav_generated.json
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/wav_gen/${s}_stage431_dur${duration}_bsz${inference_batch_size}_r${rank}_ns${nshard}_t${temperature}
    mkdir -p ${dst_dir}
    echo "filter lambda: ${filter_lambda}"
    python $VOC_DIR/generate_v3.3.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size,num_workers=6" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --length-filter="${filter_lambda}" \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --ignore-list ${ignore_list}
      #  "lambda x: x<=13" \
fi

if [ ${stage} -le 44 ] && [ ${stop_stage} -ge 44 ]; then
    echo "stage 44: Synthesize one batch waveforms from Hubert features using mismatched speaker (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_stage44.1_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.2_nvvspk.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=${n_batch} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 441 ] && [ ${stop_stage} -ge 441 ]; then
    echo "stage 441: Synthesize massive batches of waveforms from wav2vec features using mismatched speaker (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/wav_gen/${s}_stage441_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}

	echo dump_org_dir: ${dump_org_dir}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.2_nvvspk.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --max-token ${max_token}

fi

if [ ${stage} -le 442 ] && [ ${stop_stage} -ge 442 ]; then
    echo "stage 442: Synthesize one batch waveforms from wav2vec features using mismatched speaker from lsp (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    # dst_dir=$expdir/generated/$name/${s}_stage44.1_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}
    dst_dir=$expdir/generated/$name/${s}_stage442_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}
	dump_org_dir="wavenet-r9y9/egs/mulaw256_lsp960_16khz_hubert/dump/lsp/logmelspectrogram/org"
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.2_nvvspk.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=${n_batch} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 443 ] && [ ${stop_stage} -ge 443 ]; then
    echo "stage 443: Synthesize one batch waveforms from wav2vec features using matched/mismatched speaker from lsp (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    seen_speakers=$(pwd)/seen_speakers.json
    echo "use speakers: ${use_speakers} ${seen_speakers}" 
    dst_dir=$expdir/generated/$name/${s}_stage443_mt${max_token}_us${use_speakers}_s${seed}_t${temperature}_r${rank}_nshard${nshard}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.2_nvvspk.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=${n_batch} --use-speakers=${use_speakers} --seen-speakers=${seen_speakers} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 444 ] && [ ${stop_stage} -ge 444 ]; then
    echo "stage 444: Synthesize massive batches of waveforms from wav2vec features using match/mismatched speaker (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    seen_speakers=$(pwd)/seen_speakers.json
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/wav_gen_100/${s}_stage444_mt${max_token}_us${use_speakers}_s${seed}_r${rank}_nshard${nshard}

	echo dump_org_dir: ${dump_org_dir}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.2_nvvspk.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --use-speakers=${use_speakers} --seen-speakers=${seen_speakers} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 45 ] && [ ${stop_stage} -ge 45 ]; then
    echo "stage 45: Synthesize one batch waveforms from Hubert features using matched speaker with different seed (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_stage45_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.1_nv.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=10 \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${dump_org_dir}/train_no_dev --nshard ${nshard} --rank ${rank}
fi

if [ ${stage} -le 47 ] && [ ${stop_stage} -ge 47 ]; then
    echo "stage 47: Synthesize one batch waveforms from generated w2v labels using matched speaker with different seed (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/${s}_stage47_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}_${generation_tag}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.3_nvulm.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --n_batch=${n_batch} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${km_path} --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 471 ] && [ ${stop_stage} -ge 471 ]; then
    echo "stage 471: Synthesize one batch waveforms from generated w2v labels using matched speaker with different seed (seed ${seed})"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    seen_speakers=$(pwd)/seen_speakers.json
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble_gen
    dst_dir=$expdir/generated/$name/transulm_lsp100_gen_${generation_tag}/${s}_stage471_dur${duration}_bsz${inference_batch_size}_s${seed}_t${temperature}_r${rank}_nshard${nshard}_${generation_tag}
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v4.3.1_transulm.py \
      --preset $hparams --hparams="batch_size=1,num_workers=4" --duration=${duration} \
      --seed=${seed} --temperature=${temperature} --device=${device} --use-speakers=${use_speakers} --seen-speakers=${seen_speakers} \
      --checkpoint=$eval_checkpoint --dst-dir=$dst_dir --data-root ${km_path} --nshard ${nshard} --rank ${rank} --max-token ${max_token}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis lots of waveforms from WaveNet using CPU"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    s=babble
    dst_dir=$expdir/generated/$name/${s}_dur${duration}_cpu
    python $VOC_DIR/generate.py \
      --preset $hparams --hparams="batch_size=180" --duration=${duration}\
      --total-hour=${total_generation_hour} --seed=${seed} --device=cpu \
      $eval_checkpoint $dst_dir
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis lots of waveforms from WaveNet"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    sample_rate=16000
    teacher_length=$(bc <<< "${teacher_dur}"*"${sample_rate}" | awk '{print int($1)}' ) 
    s=babble
    dst_dir=$expdir/generated/$name/lsp960_wav_td0.3/${s}_dur${duration}_bsz${inference_batch_size}_seed${seed}_tdur${teacher_dur}_v2
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v2.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size,num_workers=4" --duration=${duration} \
      --total-hour=${total_generation_hour} --seed=${seed} --temperature=${temperature} --device=${device} \
      --data-root=${dump_org_dir}/train_no_dev --checkpoint=${eval_checkpoint} --dst-dir=${dst_dir} --teacher-length=${teacher_length}
fi


if [ ${stage} -le 61 ] && [ ${stop_stage} -ge 61 ]; then
    echo "stage 6.1: Synthesis lots of waveforms from WaveNet with different speaker"
    if [ -z $eval_checkpoint ]; then
      eval_checkpoint=$expdir/checkpoint_latest.pth
    fi
    
    name=$(basename $eval_checkpoint)
    name=${name/.pth/}
    sample_rate=16000
    teacher_length=$(bc <<< "${teacher_dur}"*"${sample_rate}" | awk '{print int($1)}' ) 
    s=babble
    dst_dir=$expdir/generated/$name/${s}_dur${duration}_bsz${inference_batch_size}_seed${seed}_tdur${teacher_dur}_v2.1_ulm
    mkdir -p ${dst_dir}
    python $VOC_DIR/generate_v2.1.py \
      --preset $hparams --hparams="batch_size=$inference_batch_size,num_workers=4" --duration=${duration} \
      --total-hour=${total_generation_hour} --seed=${seed} --temperature=${temperature} --device=${device} \
      --data-root=${dump_org_dir}/train_no_dev --checkpoint=${eval_checkpoint} --dst-dir=${dst_dir} --teacher-length=${teacher_length}
fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Evaluate generated waveforms"
    LM="models/4-gram.bin"
    WAV2VEC="models/wav2vec_big_960h.pt"
    INFER_SCRIPT="fairseq-wavenet-r9y9/examples/speech_recognition/infer.py"
    MANIFEST_SCRIPT="wavenet-r9y9/scripts/wav2vec_manifest.py"
    AUDIO_DIR=${audio_dir}
    MANIFEST_DIR="manifest/$(realpath --relative-to=exp ${AUDIO_DIR} | sed 's/\//-/g')"
    FULL_MANIFEST_TSV="metadata/LibriSpeech960/all.tsv"
    FULL_MANIFEST_LTR="metadata/LibriSpeech960/all.ltr"
    DICT="metadata/LibriSpeech960/dict.ltr.txt"
    RESULT_PATH="${MANIFEST_DIR}/results"
    export PATH=workplace/SCTK/bin:${PATH}
    SUBSET=train
    mkdir -p ${MANIFEST_DIR}
    mkdir -p ${RESULT_PATH}
    echo "step 7.1 creating manifest for generated audios..."
    python ${MANIFEST_SCRIPT} ${AUDIO_DIR} --dest ${MANIFEST_DIR} --ext wav --valid-percent 0 --nprocess ${nprocess}
    echo "step 7.2 find ltr for generated audios..."
    python scripts/retrieve_transcript.py --subset-tsv ${MANIFEST_DIR}/${SUBSET}.tsv --subset-ltr ${MANIFEST_DIR}/${SUBSET}.ltr \
        --full-tsv ${FULL_MANIFEST_TSV} --full-ltr ${FULL_MANIFEST_LTR}
    cp ${DICT} ${MANIFEST_DIR}
    echo "step 7.3 compute error rate of generated audios..."
    python ${INFER_SCRIPT} ${MANIFEST_DIR} --task audio_finetuning \
    --nbest 1 --path ${WAV2VEC} --gen-subset ${SUBSET} --results-path ${RESULT_PATH} --w2l-decoder kenlm \
    --lm-model ${LM} --lexicon models/lexicon_ltr.lst --lm-weight 2 --word-score -1 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 4000000 \
    --post-process letter

    sclite -r ${RESULT_PATH}/ref.units-$(basename ${WAV2VEC})-train.txt trn -h ${RESULT_PATH}/hypo.units-$(basename ${WAV2VEC})-train.txt trn -i wsj | tee ${RESULT_PATH}/uer.txt
    sclite -r ${RESULT_PATH}/ref.word-$(basename ${WAV2VEC})-train.txt trn -h ${RESULT_PATH}/hypo.word-$(basename ${WAV2VEC})-train.txt trn -i wsj | tee ${RESULT_PATH}/wer.txt
fi
