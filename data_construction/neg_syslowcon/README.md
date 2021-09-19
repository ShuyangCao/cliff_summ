# SysLowCon

-------

##### Generate

First, download the BART models pre-trained on XSum (`bart.large.xsum`) and CNN/DM (`bart.large.cnn`) from [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/bart)
and set the corresponding environment variables for them:

```shell
export XSUM_BART=/path/to/bart.large.xsum/model.pt
export CNNDM_BART=/path/to/bart.large.cnn/model.pt
```

Then run the following commands for generation:

```shell
# XSum
cd syslowcon_scripts
./xsum_train.sh
./xsum_valid.sh
python get_output.py $DATA/processed_data/syslowcon_xsum_generation/generate-train.txt \
 $DATA/processed_data/syslowcon_xsum_generation/train.jsonl
python get_output.py $DATA/processed_data/syslowcon_xsum_generation/generate-valid.txt \
 $DATA/processed_data/syslowcon_xsum_generation/valid.jsonl
 
# CNN/DM
cd syslowcon_scripts
./cnndm_train.sh
./cnndm_valid.sh
python get_output.py $DATA/processed_data/syslowcon_cnndm_generation/generate-train.txt \
 $DATA/processed_data/syslowcon_cnndm_generation/train.jsonl
python get_output.py $DATA/processed_data/syslowcon_cnndm_generation/generate-valid.txt \
 $DATA/processed_data/syslowcon_cnndm_generation/valid.jsonl
```

##### Obtain low confidence samples

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/negative_syslowcon
python filter_out.py $DATA/processed_data/syslowcon_xsum_generation/train.jsonl \
 $DATA/xsum_synthetic/negative_syslowcon/train
python filter_out.py $DATA/processed_data/syslowcon_xsum_generation/valid.jsonl \
 $DATA/xsum_synthetic/negative_syslowcon/valid

# CNN/DM
mkdir -p $DATA/cnndm_synthetic/negative_syslowcon
python filter_out.py $DATA/processed_data/syslowcon_cnndm_generation/train.jsonl \
 $DATA/cnndm_synthetic/negative_syslowcon/train
python filter_out.py $DATA/processed_data/syslowcon_cnndm_generation/valid.jsonl \
 $DATA/cnndm_synthetic/negative_syslowcon/valid
```
