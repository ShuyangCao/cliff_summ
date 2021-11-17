# CLIFF

Code for EMNLP 2021 paper "CLIFF: Contrastive Learning for Improving Faithfulness and Factuality in Abstractive Summarization"

---------

## Data Construction

For data construction, please refer to [data_construction](data_construction).
Constructed datasets are also available in [Google Drive](https://drive.google.com/drive/folders/1b7JD419DBJv2BrNduBYOs8floP1JgO0-?usp=sharing).

---------

## Training

The following scripts require that your `$DATA` folder is organized the same as the `data` folder
in [Google Drive](https://drive.google.com/drive/folders/1b7JD419DBJv2BrNduBYOs8floP1JgO0-?usp=sharing).

#### BART

Our experiments with BART use [Fairseq](https://github.com/pytorch/fairseq) at commit `0db28cd`. Newer versions might also work.
Please download the pre-trained BART model [here](https://github.com/pytorch/fairseq/tree/master/examples/bart)
and set `BART_PATH` to the downloaded model:

```shell
export BART_PATH=/path/to/bart/model.pt
```

##### Single Negative Strategy

The following command trains the models with negative samples constructed by `SysLowCon`.
It saves the trained models in `$TRAINED_MODELS/xsum/syslowcon` and `$TRAINED_MODELS/cnndm/syslowcon`.
Please change `$DATA/xsum_synthetic/negative_syslowcon` to other negative samples to train the corresponding models.

```shell
# XSum
cd scripts/bart
CUDA_VISIBLE_DEVICES=0,1 ./train_xsum_single_neg.sh \
  $DATA/xsum_synthetic/negative_syslowcon $TRAINED_MODELS/bart_xsum/syslowcon

# CNN/DM
cd scripts/bart
CUDA_VISIBLE_DEVICES=0,1 ./train_cnndm_single_neg.sh \
  $DATA/cnndm_synthetic/negative_syslowcon $TRAINED_MODELS/bart_cnndm/syslowcon
```

##### Multiple Negative Strategies

The following command trains the models with negative samples constructed by `SysLowCon` and `SwapEnt`.
It saves the trained models in `$TRAINED_MODELS/xsum/syslowcon_swapent` and `$TRAINED_MODELS/cnndm/syslowcon_swapent`.

```shell
# XSum
cd scripts/bart
CUDA_VISIBLE_DEVICES=0,1 ./train_xsum_mutli_neg.sh \
  "$DATA/xsum_synthetic/negative_syslowcon $DATA/xsum_synthetic/negative_swapent" \
  $TRAINED_MODELS/bart_xsum/syslowcon_swapent

# CNN/DM
cd scripts/bart
CUDA_VISIBLE_DEVICES=0,1 ./train_cnndm_multi_neg.sh \
  "$DATA/cnndm_synthetic/negative_syslowcon $DATA/cnndm_synthetic/negative_swapent" \
  $TRAINED_MODELS/bart_cnndm/syslowcon_swapent
```

#### Pegasus

Our experiments with Pegasus use [Huggingface Transformers](https://github.com/huggingface/transformers) `4.5.1`.
Newer versions might also work.

##### Single Negative Strategy

```shell
# XSum
cd scripts/pegasus
CUDA_VISIBLE_DEVICES=0,1 ./train_xsum_single_neg.sh \
  $DATA/xsum_synthetic/negative_syslowcon $TRAINED_MODELS/pegasus_xsum/syslowcon
  
# CNN/DM
cd scripts/pegasus
CUDA_VISIBLE_DEVICES=0,1 ./train_cnndm_single_neg.sh \
  $DATA/cnndm_synthetic/negative_syslowcon $TRAINED_MODELS/pegasus_cnndm/syslowcon
```

## Decoding

The following examples show how to decode trained models. Model checkpoints are available in 
[Google Drive](https://drive.google.com/drive/folders/1b7JD419DBJv2BrNduBYOs8floP1JgO0-?usp=sharing).

#### BART

```shell
# XSum
cd scripts/bart
./decode_xsum.sh $TRAINED_MODELS/bart_xsum/syslowcon/checkpoint_last.pt /path/to/save/dir

# CNN/DM
cd scripts/bart
./decode_cnndm.sh $TRAINED_MODELS/bart_cnndm/syslowcon/checkpoint_last.pt /path/to/save/dir
```

#### Pegasus

```shell
# XSum
cd scripts/pegasus
python run_generation.py $DATA/xsum_raw/test.source $TRAINED_MODELS/pegasus_xsum/syslowcon /path/to/save/dir

# CNN/DM
cd scripts/pegasus
python run_generation.py $DATA/cnndm_raw/test.source $TRAINED_MODELS/pegasus_cnndm/syslowcon /path/to/save/dir
```
