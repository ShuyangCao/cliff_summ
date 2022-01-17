## New Implementation

We provide a new implementation for newer versions of Fairseq that adopt Hydra Config.
The new implementation also features easier data preprocessing.

### Data Preprocessing

#### Raw Files

To preprocess data for the new implementation, please have the following files ready (for example, in `$DATA/xxx_dataset/raw`):

```
train.source
train.positive
train.positive.index
train.negative
train.negative.index
valid.source
valid.positive
valid.positive.index
valid.negative
valid.negative.index
```

- `{SPLIT}.source`: `i`-th line of the file contains the source document of the `i`-th sample.
- `{SPLIT}.positive`: each line of the file contains a positive target.
- `{SPLIT}.positive.index`: `i`-th line of the file contains the line indices for positive targets of `i`-th sample.
For example, if the `i`-th line of `{SPLIT}.positive.index` is `3 4 5`, then line 3, 4, and 5 of `{SPLIT}.positive` are positive targets of the `i`-th sample.
**Note:** the positive sample corresponding to the **last index** (e.g., `5` in the above example) of each line is used as the cross entropy target.
- `{SPLIT}.negative`: each line of the file contains a negative target.
- `{SPLIT}.negative.index`: `i`-th line of the file contains the line indices for negative targets of `i`-th sample.

#### BPE Conversion and Dataset Binarization

Assume `$DATA/pretrain_language_models/fairseq.gpt2` contains the bpe encoder file (`encoder.json`),
the bpe vocab (`vocab.bpe`), and the dictionary (`dict.txt`) you download following the 
[instructions](https://github.com/pytorch/fairseq/blob/main/examples/bart/README.summarization.md#2-bpe-preprocess) by Fairseq.

##### BPE Conversion

```shell
for SPLIT in train valid
do
    for LANG in source positive negative
    do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json $DATA/pretrain_language_models/fairseq.gpt2/encoder.json  \
    --vocab-bpe $DATA/pretrain_language_models/fairseq.gpt2/vocab.bpe \
    --inputs $DATA/xxx_dataset/raw/${SPLIT}.${LANG} \
    --outputs $DATA/xxx_dataset/raw/${SPLIT}.bpe.${LANG} \
    --workers 60 \
    --keep-empty
    done
done
```

##### Dataset Binarization

```shell
for LANG in source positive negative
do
fairseq-preprocess --only-source \
  --trainpref $DATA/xxx_dataset/raw/train.bpe.${LANG} \
  --validpref $DATA/xxx_dataset/raw/valid.bpe.${LANG} \
  --destdir $DATA/xxx_dataset/cl-bin/${LANG} \
  --workers 60 \
  --srcdict $DATA/pretrain_language_models/fairseq.gpt2/dict.txt
done

for SPLIT in train valid
do
    for LANG in positive negative
    do
    cp $DATA/xxx_dataset/raw/${SPLIT}.${LANG}.index $DATA/xxx_dataset/cl-bin/${LANG}/${SPLIT}.mapping
    done
done
```

### Training

Example Training Script:

```shell
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=8
BART_PATH=${DATA}/pretrain_language_models/bart.large/model.pt
DATA_PATH=${DATA}/xxx_dataset/cl-bin
SAVE_DIR=${DATA}/trained_models/xxx_dataset_cl/a_cl_model
USER_DIR=./model

fairseq-train $DATA_PATH --save-dir $SAVE_DIR \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS --mlp 1024 --max-neg-samples 3 \
    --task contrastive_translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch contrastive_bart_large \
    --criterion contrastive_loss --alpha 0.5 \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-update $TOTAL_NUM_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints --no-save-optimizer-state \
    --log-interval 1 \
    --find-unused-parameters \
    --user-dir $USER_DIR;
```

### Testing

Example Testing Script:

```shell
BEAM_SIZE=4
MAX_LEN_B=140
MIN_LEN=55
LEN_PEN=2.0

DATA_PATH=${DATA}/xxx_dataset/bin
MODEL_PATH=${DATA}/trained_models/xxx_dataset_cl/a_cl_model/checkpoint_best.pt
RESULT_PATH=${DATA}/decode_output/xxx_dataset_cl/a_cl_model
USER_DIR=./model

fairseq-generate $DATA_PATH \
   --path $MODEL_PATH --results-path $RESULT_PATH --task translation \
   --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
   --no-repeat-ngram-size 3 \
   --batch-size 16 --fp16 \
   --truncate-source --user-dir $USER_DIR;
```

**Note:** `DATA_PATH` here should point to the non-CL binarized dataset.