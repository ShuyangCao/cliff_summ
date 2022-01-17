BEAM_SIZE=4
MAX_LEN_B=140
MIN_LEN=55
LEN_PEN=2.0

DATA_PATH=$DATA/cnndm_binarized
MODEL_PATH=$1
RESULT_PATH=$2
USER_DIR=../../models/bart


fairseq-generate $DATA_PATH \
    --path $MODEL_PATH --results-path $RESULT_PATH \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN \
    --no-repeat-ngram-size 3 \
    --batch-size 32 --fp16 \
    --truncate-source --user-dir $USER_DIR;

python convert_bart.py --generate-dir $RESULT_PATH
