BEAM_SIZE=6
MAX_LEN_B=60
MIN_LEN=10
LEN_PEN=1.0

DATA_PATH=$DATA/xsum_binarzied
RESULT_PATH=$DATA/processed_data/syslowcon_xsum_generation


fairseq-generate $DATA_PATH \
    --path $XSUM_BART --results-path $RESULT_PATH --task translation \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN --nbest $BEAM_SIZE \
    --no-repeat-ngram-size 3 --skip-invalid-size-inputs-valid-test \
    --batch-size 16 --fp16 \
    --truncate-source --gen-subset valid;