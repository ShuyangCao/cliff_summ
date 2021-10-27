# Mask Filling / Regeneration

We use [Stanza](https://stanfordnlp.github.io/stanza/index.html) `1.2` and [spacy-stanza](https://github.com/explosion/spacy-stanza) `1.0.0`
for parsing.

```shell
# create virtual environment (for example, conda)
conda create -n mask_regen python=3.8
conda activate mask_regen
pip install -r requirements.txt

# in case stanza model is not downloaded
python
>> import stanza
>> stanza.download('en')
```

--------

### Parse original document and summary

```shell
# XSum
mkdir -p $DATA/processed_data/xsum_stanza_docbin
python detect_relation_ne_document_summary.py $DATA/xsum_raw/train.source $DATA/xsum_raw/train.target \
 $DATA/processed_data/xsum_stanza_docbin/train
python detect_relation_ne_document_summary.py $DATA/xsum_raw/validation.source $DATA/xsum_raw/validation.target \
 $DATA/processed_data/xsum_stanza_docbin/valid

mkdir -p $DATA/processed_data/xsum_relation
python docbin2relation.py $DATA/processed_data/xsum_stanza_docbin/train.target \
 $DATA/processed_data/xsum_relation/train.jsonl
python docbin2relation.py $DATA/processed_data/xsum_stanza_docbin/valid.target \
 $DATA/processed_data/xsum_relation/valid.jsonl
 
# CNN/DM
mkdir -p $DATA/processed_data/cnndm_stanza_docbin
python detect_relation_ne_document_summary.py $DATA/cnndm_raw/train.source $DATA/cnndm_raw/train.target \
 $DATA/processed_data/cnndm_stanza_docbin/train
python detect_relation_ne_document_summary.py $DATA/cnndm_raw/val.source $DATA/cnndm_raw/val.target \
 $DATA/processed_data/cnndm_stanza_docbin/valid

mkdir -p $DATA/processed_data/cnndm_relation
python docbin2relation.py $DATA/processed_data/cnndm_stanza_docbin/train.target \
 $DATA/processed_data/cnndm_relation/train.jsonl
python docbin2relation.py $DATA/processed_data/cnndm_stanza_docbin/valid.target \
 $DATA/processed_data/cnndm_relation/valid.jsonl
```

### Mask filling

##### Fill with BART

```shell
# XSum
mkdir -p $DATA/processed_data/xsum_mask_filling
python lm_mask_fill.py $DATA/processed_data/xsum_relation/train.jsonl \
 $DATA/processed_data/xsum_mask_filling/train_bart_fill.jsonl
python lm_mask_fill.py $DATA/processed_data/xsum_relation/valid.jsonl \
 $DATA/processed_data/xsum_mask_filling/valid_bart_fill.jsonl

# CNN/DM
mkdir -p $DATA/processed_data/cnndm_mask_filling
python lm_mask_fill.py $DATA/processed_data/cnndm_relation/train.jsonl \
 $DATA/processed_data/cnndm_mask_filling/train_bart_fill.jsonl
python lm_mask_fill.py $DATA/processed_data/cnndm_relation/valid.jsonl \
 $DATA/processed_data/cnndm_mask_filling/valid_bart_fill.jsonl
```

##### Parse filled samples

```shell
# XSum
python generated_json_to_text.py $DATA/processed_data/xsum_relation/train.jsonl \
 $DATA/processed_data/xsum_mask_filling/train_bart_fill.jsonl \
 $DATA/processed_data/xsum_mask_filling/train_generated.txt \
 $DATA/processed_data/xsum_mask_filling/train_generated.other
python generated_json_to_text.py $DATA/processed_data/xsum_relation/valid.jsonl \
 $DATA/processed_data/xsum_mask_filling/valid_bart_fill.jsonl \
 $DATA/processed_data/xsum_mask_filling/valid_generated.txt \
 $DATA/processed_data/xsum_mask_filling/valid_generated.other

python detect_relation_ne_summary.py $DATA/processed_data/xsum_mask_filling/train_generated.txt \
 $DATA/processed_data/xsum_mask_filling/train_generated.doc
python detect_relation_ne_summary.py $DATA/processed_data/xsum_mask_filling/valid_generated.txt \
 $DATA/processed_data/xsum_mask_filling/valid_generated.doc
 
# CNN/DM
python generated_json_to_text.py $DATA/processed_data/cnndm_relation/train.jsonl \
 $DATA/processed_data/cnndm_mask_filling/train_bart_fill.jsonl \
 $DATA/processed_data/cnndm_mask_filling/train_generated.txt \
 $DATA/processed_data/cnndm_mask_filling/train_generated.other
python generated_json_to_text.py $DATA/processed_data/cnndm_relation/valid.jsonl \
 $DATA/processed_data/cnndm_mask_filling/valid_bart_fill.jsonl \
 $DATA/processed_data/cnndm_mask_filling/valid_generated.txt \
 $DATA/processed_data/cnndm_mask_filling/valid_generated.other

python detect_relation_ne_summary.py $DATA/processed_data/cnndm_mask_filling/train_generated.txt \
 $DATA/processed_data/cnndm_mask_filling/train_generated.doc
python detect_relation_ne_summary.py $DATA/processed_data/cnndm_mask_filling/valid_generated.txt \
 $DATA/processed_data/cnndm_mask_filling/valid_generated.doc
```

##### Filter filled samples

```shell
# XSum
python filter_generated.py --generated-docbins $DATA/processed_data/xsum_mask_filling/train_generated.doc \
 --source-docbins $DATA/processed_data/xsum_stanza_docbin/train.source \
 --target-docbins $DATA/processed_data/xsum_stanza_docbin/train.target \
 --other $DATA/processed_data/xsum_mask_filling/train_generated.other \
 $DATA/processed_data/xsum_mask_filling/train_filtered.jsonl
python filter_generated.py --generated-docbins $DATA/processed_data/xsum_mask_filling/valid_generated.doc \
 --source-docbins $DATA/processed_data/xsum_stanza_docbin/valid.source \
 --target-docbins $DATA/processed_data/xsum_stanza_docbin/valid.target \
 --other $DATA/processed_data/xsum_mask_filling/valid_generated.other \
 $DATA/processed_data/xsum_mask_filling/valid_filtered.jsonl

# CNN/DM
python filter_generated.py --generated-docbins $DATA/processed_data/cnndm_mask_filling/train_generated.doc \
 --source-docbins $DATA/processed_data/cnndm_stanza_docbin/train.source \
 --target-docbins $DATA/processed_data/cnndm_stanza_docbin/train.target \
 --other $DATA/processed_data/cnndm_mask_filling/train_generated.other \
 $DATA/processed_data/cnndm_mask_filling/train_filtered.jsonl
python filter_generated.py --generated-docbins $DATA/processed_data/cnndm_mask_filling/valid_generated.doc \
 --source-docbins $DATA/processed_data/cnndm_stanza_docbin/valid.source \
 --target-docbins $DATA/processed_data/cnndm_stanza_docbin/valid.target \
 --other $DATA/processed_data/cnndm_mask_filling/valid_generated.other \
 $DATA/processed_data/cnndm_mask_filling/valid_filtered.jsonl
```

##### Convert format

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/negative_maskent
mkdir -p $DATA/xsum_synthetic/negative_maskrel
python get_new_ent_out.py $DATA/processed_data/xsum_mask_filling/train_filtered.jsonl $DATA/xsum_synthetic/negative_maskent/train
python get_new_rel_out.py $DATA/processed_data/xsum_mask_filling/train_filtered.jsonl $DATA/xsum_synthetic/negative_maskrel/train
python get_new_ent_out.py $DATA/processed_data/xsum_mask_filling/valid_filtered.jsonl $DATA/xsum_synthetic/negative_maskent/valid
python get_new_rel_out.py $DATA/processed_data/xsum_mask_filling/valid_filtered.jsonl $DATA/xsum_synthetic/negative_maskrel/valid

# CNN/DM
mkdir -p $DATA/cnndm_synthetic/negative_maskent
mkdir -p $DATA/cnndm_synthetic/negative_maskrel
python get_new_ent_out.py $DATA/processed_data/cnndm_mask_filling/train_filtered.jsonl $DATA/cnndm_synthetic/negative_maskent/train
python get_new_rel_out.py $DATA/processed_data/cnndm_mask_filling/train_filtered.jsonl $DATA/cnndm_synthetic/negative_maskrel/train
python get_new_ent_out.py $DATA/processed_data/cnndm_mask_filling/valid_filtered.jsonl $DATA/cnndm_synthetic/negative_maskent/valid
python get_new_rel_out.py $DATA/processed_data/cnndm_mask_filling/valid_filtered.jsonl $DATA/cnndm_synthetic/negative_maskrel/valid
```

##### Convert BPE

```shell
# XSum
for SPLIT in train valid
do
  for STG in maskent maskrel
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json $DATA/fairseq.gpt2/encoder.json  \
            --vocab-bpe $DATA/fairseq.gpt2/vocab.bpe \
            --inputs "$DATA/xsum_synthetic/negative_$STG/$SPLIT.raw_target" \
            --outputs "$DATA/xsum_synthetic/negative_$STG/$SPLIT.neg_target" \
            --workers 60 \
            --keep-empty;
  done
done

# CNN/DM
for SPLIT in train valid
do
  for STG in maskent maskrel
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json $DATA/fairseq.gpt2/encoder.json  \
            --vocab-bpe $DATA/fairseq.gpt2/vocab.bpe \
            --inputs "$DATA/cnndm_synthetic/negative_$STG/$SPLIT.raw_target" \
            --outputs "$DATA/cnndm_synthetic/negative_$STG/$SPLIT.neg_target" \
            --workers 60 \
            --keep-empty;
  done
done
```

### Regeneration

##### Create regeneration data

```shell
# XSum
mkdir -p $DATA/processed_data/xsum_regeneration
python regeneration_data.py $DATA/xsum_raw/train.bpe.source \
 $DATA/processed_data/xsum_relation/train.jsonl \
 $DATA/processed_data/xsum_regeneration/train
python regeneration_data.py $DATA/xsum_raw/validation.bpe.source \
 $DATA/processed_data/xsum_relation/valid.jsonl \
 $DATA/processed_data/xsum_regeneration/valid

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/processed_data/xsum_regeneration/train.bpe \
 --validpref $DATA/processed_data/xsum_regeneration/valid.bpe \
 --destdir $DATA/processed_data/xsum_regeneration \
 --workers 60 \
 --srcdict $DATA/fairseq.gpt2/dict.txt \
 --tgtdict $DATA/fairseq.gpt2/dict.txt
 
# CNN/DM
mkdir -p $DATA/processed_data/cnndm_regeneration
python regeneration_data.py $DATA/cnndm_raw/train.bpe.source \
 $DATA/processed_data/cnndm_relation/train.jsonl \
 $DATA/processed_data/cnndm_regeneration/train
python regeneration_data.py $DATA/cnndm_raw/valid.bpe.source \
 $DATA/processed_data/cnndm_relation/valid.jsonl \
 $DATA/processed_data/cnndm_regeneration/valid

fairseq-preprocess --source-lang source --target-lang target \
 --trainpref $DATA/processed_data/cnndm_regeneration/train.bpe \
 --validpref $DATA/processed_data/cnndm_regeneration/valid.bpe \
 --destdir $DATA/processed_data/cnndm_regeneration \
 --workers 60 \
 --srcdict $DATA/fairseq.gpt2/dict.txt \
 --tgtdict $DATA/fairseq.gpt2/dict.txt
```

##### Regenerate

First, download the BART models pre-trained on XSum (`bart.large.xsum`) and CNN/DM (`bart.large.cnn`) from [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/bart)
and set the corresponding environment variables for them:

```shell
export XSUM_BART=/path/to/bart.large.xsum/model.pt
export CNNDM_BART=/path/to/bart.large.cnn/model.pt
```

Then run the following commands for regeneration:

```shell
# XSum
cd regen_scripts
./regen_xsum_train.sh
./regen_xsum_valid.sh
mkdir -p $DATA/processed_data/xsum_regeneration_output
python convert_regeneration.py --generate-dir $DATA/processed_data/xsum_regeneration_output
python add_prompt.py $DATA/processed_data/xsum_regeneration/train.bpe.target \
  $DATA/processed_data/xsum_regeneration_output/formatted-train.txt \
  $DATA/processed_data/xsum_regeneration/train.other \
  $DATA/processed_data/xsum_regeneration_output/train_generated.txt \
  $DATA/processed_data/xsum_regeneration_output/train_generated.other
python add_prompt.py $DATA/processed_data/xsum_regeneration/valid.bpe.target \
  $DATA/processed_data/xsum_regeneration_output/formatted-valid.txt \
  $DATA/processed_data/xsum_regeneration/valid.other \
  $DATA/processed_data/xsum_regeneration_output/valid_generated.txt \
  $DATA/processed_data/xsum_regeneration_output/valid_generated.other

# CNN/DM
cd regen_scripts
./regen_cnndm_train.sh
./regen_cnndm_valid.sh
mkdir -p $DATA/processed_data/cnndm_regeneration_output
python convert_regeneration.py --generate-dir $DATA/processed_data/cnndm_regeneration_output
python add_prompt.py $DATA/processed_data/cnndm_regeneration/train.bpe.target \
  $DATA/processed_data/cnndm_regeneration_output/formatted-train.txt \
  $DATA/processed_data/cnndm_regeneration/train.other \
  $DATA/processed_data/cnndm_regeneration_output/train_generated.txt \
  $DATA/processed_data/cnndm_regeneration_output/train_generated.other
python add_prompt.py $DATA/processed_data/cnndm_regeneration/valid.bpe.target \
  $DATA/processed_data/cnndm_regeneration_output/formatted-valid.txt \
  $DATA/processed_data/cnndm_regeneration/valid.other \
  $DATA/processed_data/cnndm_regeneration_output/valid_generated.txt \
  $DATA/processed_data/cnndm_regeneration_output/valid_generated.other
```

##### Parse regenerated samples

```shell
# XSum
python detect_relation_ne_summary.py $DATA/processed_data/xsum_regeneration_output/train_generated.txt \
 $DATA/processed_data/xsum_regeneration_output/train_generated.doc
python detect_relation_ne_summary.py $DATA/processed_data/xsum_regeneration_output/valid_generated.txt \
 $DATA/processed_data/xsum_regeneration_output/valid_generated.doc
 
# CNN/DM
python detect_relation_ne_summary.py $DATA/processed_data/cnndm_regeneration_output/train_generated.txt \
 $DATA/processed_data/cnndm_regeneration_output/train_generated.doc
python detect_relation_ne_summary.py $DATA/processed_data/cnndm_regeneration_output/valid_generated.txt \
 $DATA/processed_data/cnndm_regeneration_output/valid_generated.doc
```

##### Filter regenerated samples

```shell
# XSum
python filter_generated.py --generated-docbins $DATA/processed_data/xsum_regeneration_output/train_generated.doc \
 --source-docbins $DATA/processed_data/xsum_stanza_docbin/train.source \
 --target-docbins $DATA/processed_data/xsum_stanza_docbin/train.target \
 --other $DATA/processed_data/xsum_regeneration_output/train_generated.other \
 $DATA/processed_data/xsum_regeneration_output/train_filtered.jsonl
python filter_generated.py --generated-docbins $DATA/processed_data/xsum_regeneration_output/valid_generated.doc \
 --source-docbins $DATA/processed_data/xsum_stanza_docbin/valid.source \
 --target-docbins $DATA/processed_data/xsum_stanza_docbin/valid.target \
 --other $DATA/processed_data/xsum_regeneration_output/valid_generated.other \
 $DATA/processed_data/xsum_regeneration_output/valid_filtered.jsonl

# CNN/DM
python filter_generated.py --generated-docbins $DATA/processed_data/cnndm_regeneration_output/train_generated.doc \
 --source-docbins $DATA/processed_data/cnndm_stanza_docbin/train.source \
 --target-docbins $DATA/processed_data/cnndm_stanza_docbin/train.target \
 --other $DATA/processed_data/cnndm_regeneration_output/train_generated.other \
 $DATA/processed_data/cnndm_regeneration_output/train_filtered.jsonl
python filter_generated.py --generated-docbins $DATA/processed_data/cnndm_regeneration_output/valid_generated.doc \
 --source-docbins $DATA/processed_data/cnndm_stanza_docbin/valid.source \
 --target-docbins $DATA/processed_data/cnndm_stanza_docbin/valid.target \
 --other $DATA/processed_data/cnndm_regeneration_output/valid_generated.other \
 $DATA/processed_data/cnndm_regeneration_output/valid_filtered.jsonl
```

##### Convert format

```shell
# XSum
mkdir -p $DATA/xsum_synthetic/negative_regenent
mkdir -p $DATA/xsum_synthetic/negative_regenrel
python get_new_ent_out.py $DATA/processed_data/xsum_regeneration_output/train_filtered.jsonl $DATA/xsum_synthetic/negative_regenent/train
python get_new_rel_out.py $DATA/processed_data/xsum_regeneration_output/train_filtered.jsonl $DATA/xsum_synthetic/negative_regenrel/train
python get_new_ent_out.py $DATA/processed_data/xsum_regeneration_output/valid_filtered.jsonl $DATA/xsum_synthetic/negative_regenent/valid
python get_new_rel_out.py $DATA/processed_data/xsum_regeneration_output/valid_filtered.jsonl $DATA/xsum_synthetic/negative_regenrel/valid

# CNN/DM
mkdir -p $DATA/cnndm_synthetic/negative_regenent
mkdir -p $DATA/cnndm_synthetic/negative_regenrel
python get_new_ent_out.py $DATA/processed_data/cnndm_regeneration_output/train_filtered.jsonl $DATA/cnndm_synthetic/negative_regenent/train
python get_new_rel_out.py $DATA/processed_data/cnndm_regeneration_output/train_filtered.jsonl $DATA/cnndm_synthetic/negative_regenrel/train
python get_new_ent_out.py $DATA/processed_data/cnndm_regeneration_output/valid_filtered.jsonl $DATA/cnndm_synthetic/negative_regenent/valid
python get_new_rel_out.py $DATA/processed_data/cnndm_regeneration_output/valid_filtered.jsonl $DATA/cnndm_synthetic/negative_regenrel/valid
```

##### Convert BPE

```shell
# XSum
for SPLIT in train valid
do
  for STG in regenent regenrel
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json $DATA/fairseq.gpt2/encoder.json  \
            --vocab-bpe $DATA/fairseq.gpt2/vocab.bpe \
            --inputs "$DATA/xsum_synthetic/negative_$STG/$SPLIT.raw_target" \
            --outputs "$DATA/xsum_synthetic/negative_$STG/$SPLIT.neg_target" \
            --workers 60 \
            --keep-empty;
  done
done

# CNN/DM
for SPLIT in train valid
do
  for STG in regenent regenrel
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json $DATA/fairseq.gpt2/encoder.json  \
            --vocab-bpe $DATA/fairseq.gpt2/vocab.bpe \
            --inputs "$DATA/cnndm_synthetic/negative_$STG/$SPLIT.raw_target" \
            --outputs "$DATA/cnndm_synthetic/negative_$STG/$SPLIT.neg_target" \
            --workers 60 \
            --keep-empty;
  done
done
```
