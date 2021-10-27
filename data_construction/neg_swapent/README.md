# SwapEnt

Our `SwapEnt` strategy requires [neuralcoref](https://github.com/huggingface/neuralcoref), 
which is only compatible with SpaCy 2. Please make sure you have a separate environment for `SwapEnt`,
as the other codes are based on SpaCy 3.

```shell
# create virtual environment (for example, conda)
conda create -n swapent python=3.8
conda activate swapent
pip install -r requirements.txt

# neuralcoref should install from source
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -e .
```

-------

#### Swap entity

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/negative_swapent
python swap_entity.py $DATA/xsum_raw/train.source $DATA/xsum_raw/train.target \
 $DATA/xsum_synthetic/negative_swapent/train_swap_same_entity.jsonl
python swap_entity.py $DATA/xsum_raw/validation.source $DATA/xsum_raw/validation.target \
 $DATA/xsum_synthetic/negative_swapent/valid_swap_same_entity.jsonl
 
# CNN/DM
mkdir -p $DATA/cnndm_synthetic/negative_swapent
python swap_entity.py $DATA/cnndm_raw/train.source $DATA/cnndm_raw/train.target \
 $DATA/cnndm_synthetic/negative_swapent/train_swap_same_entity.jsonl
python swap_entity.py $DATA/cnndm_raw/val.source $DATA/cnndm_raw/val.target \
 $DATA/cnndm_synthetic/negative_swapent/valid_swap_same_entity.jsonl
```

#### Convert format

```shell
# XSum
python swap_entity_out.py $DATA/xsum_raw/train.bpe.source $DATA/xsum_synthetic/negative_swapent/train_swap_same_entity.jsonl \
 $DATA/xsum_synthetic/negative_swapent/train
python swap_entity_out.py $DATA/xsum_raw/validation.bpe.source $DATA/xsum_synthetic/negative_swapent/valid_swap_same_entity.jsonl \
 $DATA/xsum_synthetic/negative_swapent/valid
 
# CNN/DM
python swap_entity_out.py $DATA/cnndm_raw/train.bpe.source $DATA/cnndm_synthetic/negative_swapent/train_swap_same_entity.jsonl \
 $DATA/cnndm_synthetic/negative_swapent/train
python swap_entity_out.py $DATA/cnndm_raw/val.bpe.source $DATA/cnndm_synthetic/negative_swapent/valid_swap_same_entity.jsonl \
 $DATA/cnndm_synthetic/negative_swapent/valid
```
