NEG_DATA=$1
OUTPUT_DIR=$2

cd ../../models/pegasus
python -m torch.distributed.launch --nproc_per_node=2 contrastive_train.py \
  --ori_data $DATA/xsum_raw \
  --pos_data $DATA/xsum_synthetic/positive_bt_filter \
  --neg_data $NEG_DATA \
  --output_dir $OUTPUT_DIR \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --max_input_length 1024 \
  --max_target_length 64 \
  --max_neg_samples 5 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --save_total_limit 3 \
  --max_steps 30000 \
  --sharded_ddp simple \
  --label_smoothing_factor 0.1 \
  --adafactor