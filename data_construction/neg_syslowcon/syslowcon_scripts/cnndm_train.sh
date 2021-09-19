BEAM_SIZE=4
MAX_LEN_B=140
MIN_LEN=55
LEN_PEN=2.0

DATA_PATH=$DATA/cnndm_binarized
RESULT_PATH=$DATA/processed_data/syslowcon_cnndm_generation


fairseq-generate $DATA_PATH \
    --path $CNNDM_BART --results-path $RESULT_PATH --task translation \
    --beam $BEAM_SIZE --max-len-b $MAX_LEN_B --min-len $MIN_LEN --lenpen $LEN_PEN --nbest $BEAM_SIZE \
    --no-repeat-ngram-size 3 --skip-invalid-size-inputs-valid-test \
    --batch-size 16 --fp16 \
    --truncate-source --gen-subset train;