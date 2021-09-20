# Back Translation

-------

We use [nlpaug](https://github.com/makcedward/nlpaug/) `1.1.3`. Please follow their installation instruction.
**Note: nlpaug switch to transformers for their back translation model in the newer version, which is not compatible with our code.**

Back translated samples with novel entities are discarded. 
We use the Wikipedia [demonym list](https://en.wikipedia.org/wiki/List_of_adjectival_and_demonymic_forms_for_countries_and_nations) 
and [countryinfo](https://github.com/porimol/countryinfo) library to
keep the paraphrases that exchange nations' names with their adjectival and demonymic forms.

#### Create raw positive samples

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/positive_bt
python nlpaug_bt.py $DATA/xsum_raw/train.target $DATA/xsum_synthetic/positive_bt/train
python nlpaug_bt.py $DATA/xsum_raw/validation.target $DATA/xsum_synthetic/positive_bt/valid

# CNN/DM
mkdir -p $DATA/cnndm_synthetic/positive_bt
python nlpaug_bt.py $DATA/cnndm_raw/train.target $DATA/cnndm_synthetic/positive_bt/train
python nlpaug_bt.py $DATA/cnndm_raw/val.target $DATA/cnndm_synthetic/positive_bt/valid
```

#### Filter invalid positive samples

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/positive_bt_filter
python nlpaug_bt_filter.py $DATA/xsum_synthetic/positive_bt/train.raw_target $DATA/xsum_raw/train.target $DATA/xsum_synthetic/positive_bt_filter/train
python nlpaug_bt_filter.py $DATA/xsum_synthetic/positive_bt/valid.raw_target $DATA/xsum_raw/validation.target $DATA/xsum_synthetic/positive_bt_filter/valid

# CNN/DM
mkdir -p $DATA/cnndm_synthetic/positive_bt_filter
python nlpaug_bt_filter.py $DATA/cnndm_synthetic/positive_bt/train.raw_target $DATA/cnndm_raw/train.target $DATA/cnndm_synthetic/positive_bt_filter/train
python nlpaug_bt_filter.py $DATA/cnndm_synthetic/positive_bt/valid.raw_target $DATA/cnndm_raw/val.target $DATA/cnndm_synthetic/positive_bt_filter/valid
```

#### Further processing

```shell
# XSum
python convert_and_combine.py $DATA/xsum_synthetic/positive_bt_filter/train.raw_target \
 /$DATA/xsum_raw/train.target $DATA/xsum_synthetic/positive_bt_filter/train.raw_other \
 $DATA/xsum_synthetic/positive_bt_filter/train 
python convert_and_combine.py $DATA/xsum_synthetic/positive_bt_filter/valid.raw_target \
 /$DATA/xsum_raw/validation.target $DATA/xsum_synthetic/positive_bt_filter/valid.raw_other \
 $DATA/xsum_synthetic/positive_bt_filter/valid
 
# CNN/DM
python convert_and_combine.py $DATA/cnndm_synthetic/positive_bt_filter/train.raw_target \
 /$DATA/cnndm_raw/train.target $DATA/cnndm_synthetic/positive_bt_filter/train.raw_other \
 $DATA/cnndm_synthetic/positive_bt_filter/train 
python convert_and_combine.py $DATA/cnndm_synthetic/positive_bt_filter/valid.raw_target \
 /$DATA/cnndm_raw/val.target $DATA/cnndm_synthetic/positive_bt_filter/valid.raw_other \
 $DATA/cnndm_synthetic/positive_bt_filter/valid
```
