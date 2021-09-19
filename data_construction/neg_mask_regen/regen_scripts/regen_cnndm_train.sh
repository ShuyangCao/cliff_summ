BEAM_SIZE=3
MAX_LEN_B=140
MIN_LEN=1
LEN_PEN=1.0

DATA_PATH=$DATA/processed_data/cnndm_regeneration
RESULT_PATH=$DATA/processed_data/cnndm_regeneration_output
USER_DIR=../../../models/regeneration


fairseq-generate $DATA_PATH --gen-subset train \
    --path $CNNDM_BART --model-overrides '{"arch": "temp_bart_large"}' \
    --results-path $RESULT_PATH --task regeneration --left-pad-target True \
    --sampling --sampling-topp 0.7 \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN --nbest $BEAM_SIZE \
    --batch-size 16 --fp16 --skip-invalid-size-inputs-valid-test \
    --truncate-source \
    --user-dir $USER_DIR;