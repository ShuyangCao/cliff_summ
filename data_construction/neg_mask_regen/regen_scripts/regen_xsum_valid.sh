BEAM_SIZE=10
MAX_LEN_B=60
MIN_LEN=1
LEN_PEN=1.0

DATA_PATH=$DATA/processed_data/xsum_regeneration
RESULT_PATH=$DATA/processed_data/xsum_regeneration_output
USER_DIR=../../../models/regeneration


fairseq-generate $DATA_PATH --gen-subset valid \
    --path $XSUM_BART --model-overrides '{"arch": "temp_bart_large"}' \
    --results-path $RESULT_PATH --task regeneration --left-pad-target True \
    --sampling --sampling-topp 0.7 \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN --nbest $BEAM_SIZE \
    --batch-size 32 --fp16 \
    --truncate-source \
    --user-dir $USER_DIR;
