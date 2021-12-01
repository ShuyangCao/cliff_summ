TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=32
NEG_DIR=$1
SAVE_PATH=$2
POS_DIR=$DATA/cnndm_synthetic/positive_bt_filter
DATA_DIR=$DATA/cnndm_binarized
USER_DIR=../../models/bart


fairseq-train $DATA_DIR --pos-data $POS_DIR --neg-data $NEG_DIR --max-neg-samples 4 \
    --restore-file $BART_PATH --save-dir $SAVE_PATH \
    --max-tokens $MAX_TOKENS \
    --task contrastive_translation_multi_neg --mlp 1024 \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch contrastive_bart_large \
    --criterion contrastive_loss \
    --label-smoothing 0.1 \
    --fixed-validation-seed 7 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test --max-epoch 5 \
    --no-save-optimizer-state --no-epoch-checkpoints \
    --find-unused-parameters \
    --user-dir $USER_DIR;