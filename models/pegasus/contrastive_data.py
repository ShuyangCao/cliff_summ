from torch.utils.data import Dataset
from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration
import torch
import random

import os
from collections import defaultdict
import itertools
from dataclasses import dataclass


def load_error_index(file_path):
    bpe_starts = []
    with open(file_path) as f:
        for line in f:
            tmp = line.strip().split('\t')
            pos_id = tmp[0]
            try:
                bpe_start = tmp[1]
            except IndexError:
                bpe_start = 0
            bpe_starts.append((int(pos_id), int(bpe_start)))
    return bpe_starts


def load_raw_aug_samples(file_path, bpe_starts):
    id_list = []
    aug_samples = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            tokens = line.strip()
            pos_id, bpe_start = bpe_starts[i]
            if bpe_start == -1:
                continue
            id_list.append(pos_id)
            aug_samples.append(tokens)
    return id_list, aug_samples


def load_ori(ori_source, ori_target):
    with open(ori_source) as f:
        sources = [line.strip() for line in f]
    with open(ori_target) as f:
        targets = [line.strip() for line in f]

    return sources, targets


class ContrastiveDataset(Dataset):
    def __init__(self, ori_prefix, pos_prefix, neg_prefix, max_neg_samples, max_input_length, max_target_length):
        tokenizer = PegasusTokenizerFast.from_pretrained('google/pegasus-large')

        ori_sources, ori_targets = load_ori(ori_prefix + '.source', ori_prefix + '.target')

        pos_other = load_error_index(pos_prefix + ".other")
        pos_ids, pos_samples = load_raw_aug_samples(pos_prefix + ".combine_target", pos_other)

        neg_other = load_error_index(neg_prefix + ".other")
        neg_ids, neg_samples = load_raw_aug_samples(neg_prefix + ".raw_target", neg_other)

        tokenized_sources = tokenizer(ori_sources, truncation=True, max_length=max_input_length)

        tokenized_sources = [{'input_ids': input_ids, 'attention_mask': attention_mask} for input_ids, attention_mask in zip(*tokenized_sources.values())]

        with tokenizer.as_target_tokenizer():

            tokenized_pos_samples = tokenizer(pos_samples, truncation=True, max_length=max_target_length)['input_ids']
            tokenized_neg_samples = tokenizer(neg_samples, truncation=True, max_length=max_target_length)['input_ids']

        self.tokenized_sources = tokenized_sources

        pos_sample_dict = defaultdict(list)
        for pos_id, pos_sample in zip(pos_ids, tokenized_pos_samples):
            pos_sample_dict[pos_id].append(pos_sample)

        neg_sample_dict = defaultdict(list)
        for neg_id, neg_sample in zip(neg_ids, tokenized_neg_samples):
            neg_sample_dict[neg_id].append(neg_sample)
        neg_sample_dict = {neg_id: random.sample(neg_samples, min(max_neg_samples, len(neg_samples))) for neg_id, neg_samples in neg_sample_dict.items()}

        self.tokenized_pos_samples = pos_sample_dict
        self.tokenized_neg_samples = neg_sample_dict

    def __len__(self):
        return len(self.tokenized_sources)

    def __getitem__(self, item):
        return {
            'source': self.tokenized_sources[item],
            'pos_samples': self.tokenized_pos_samples.get(item, []),
            'neg_samples': self.tokenized_neg_samples.get(item, [])
        }


@dataclass
class DataCollatorForContrastive:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`
            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PegasusTokenizerFast
    model: PegasusForConditionalGeneration = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    label_pad_token_id = -100

    def __call__(self, features):
        sources = []
        targets = []

        src_select_indices = []

        positive_contrast_pos = []
        valid_contrast_pos = []
        cross_entropy_pos = []

        accumulate = 0
        for i, feature in enumerate(features):
            valid_i = []
            valid_j = []

            source = feature['source']  # {input_ids, attention_mask}
            sources.append(source)

            for pos_target in feature['pos_samples']:
                src_select_indices.append(i)

                targets.append({'labels': pos_target})

                valid_i.append(accumulate)
                valid_j.append(accumulate)
                accumulate += 1
            cross_entropy_pos.append(accumulate - 1)

            comb = list(itertools.combinations(valid_i, 2))
            positive_contrast_pos.extend(comb)

            for neg_target in feature['neg_samples']:
                src_select_indices.append(i)

                targets.append({'labels': neg_target})

                valid_j.append(accumulate)
                accumulate += 1

            comb = list(itertools.combinations(valid_j, 2))
            valid_contrast_pos.extend(comb)

        positive_contrast = torch.zeros((len(targets), len(targets)), dtype=torch.bool)
        for aaa, bbb in positive_contrast_pos:
            positive_contrast[aaa, bbb] = True
            positive_contrast[bbb, aaa] = True

        valid_contrast = torch.zeros((len(targets), len(targets)), dtype=torch.bool)
        for aaa, bbb in valid_contrast_pos:
            valid_contrast[aaa, bbb] = True
            valid_contrast[bbb, aaa] = True

        for i in range(len(targets)):
            valid_contrast[i, i] = False
            positive_contrast[i, i] = False

        source_features = sources
        target_features = targets

        labels = [feature["labels"] for feature in target_features] if "labels" in target_features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in target_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        source_features = self.tokenizer.pad(
            source_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels = torch.tensor([feature['labels'] for feature in target_features], dtype=torch.long)

        # prepare decoder_input_ids
        decoder_input_ids = None
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)

        all_features = {
            'input_ids': source_features['input_ids'],
            'attention_mask': source_features['attention_mask'],
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            'src_select_indices': torch.tensor(src_select_indices, dtype=torch.long),
            'ce_pos': torch.tensor(cross_entropy_pos, dtype=torch.long),
            'positive_contrast': positive_contrast,
            'valid_contrast': valid_contrast
        }

        return all_features
