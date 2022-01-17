# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import itertools
import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )

    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )

        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    # to be used in index_select(0, )
    src_select_index = torch.LongTensor([i for i, s in enumerate(samples) for _ in s['pos_target'] + s['neg_target']])

    contrast_prev_output_tokens = data_utils.collate_tokens(
        [tgt for s in samples for tgt in s['pos_target'] + s['neg_target']],
        pad_idx,
        eos_idx,
        left_pad_target,
        True,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
        pad_to_multiple=pad_to_multiple,
    )

    contrast_target = data_utils.collate_tokens(
        [s['pos_target'][-1] for s in samples],
        pad_idx,
        eos_idx,
        left_pad_target,
        False,
        contrast_prev_output_tokens.size(1),
        pad_to_multiple=pad_to_multiple,
    )

    cross_entropy_pos = []
    accumulate_cnt = 0
    for s in samples:
        s_pos = len(s["pos_target"])
        accumulate_cnt += s_pos
        cross_entropy_pos.append(accumulate_cnt - 1)
        s_neg = len(s["neg_target"])
        accumulate_cnt += s_neg

    if samples[0].get("pos_ne", None) is not None:
        positive_contrast_pos = []
        valid_contrast_pos = []
        contrast_ne_tokens = []

        accumulate = 0
        for s in samples:
            valid_i = []
            valid_j = []
            for tgt_ne in s['pos_ne']:
                contrast_ne_tokens.append(tgt_ne)
                if not torch.all((tgt_ne == 0)):
                    valid_i.append(accumulate)
                    valid_j.append(accumulate)
                accumulate += 1
            comb = list(itertools.combinations(valid_i, 2))

            positive_contrast_pos.extend(comb)

            for tgt_ne in s['neg_ne']:
                contrast_ne_tokens.append(tgt_ne)
                valid_j.append(accumulate)
                accumulate += 1

            comb = list(itertools.combinations(valid_j, 2))
            valid_contrast_pos.extend(comb)

        contrast_ne_tokens = data_utils.collate_tokens(
            contrast_ne_tokens,
            0,
            eos_idx,
            left_pad_target,
            False,
            contrast_prev_output_tokens.size(1),
            pad_to_multiple=pad_to_multiple,
        )

        positive_contrast = torch.zeros((contrast_ne_tokens.size(0), contrast_ne_tokens.size(0)), dtype=torch.bool)
        for aaa, bbb in positive_contrast_pos:
            positive_contrast[aaa, bbb] = True
            positive_contrast[bbb, aaa] = True

        valid_contrast = torch.zeros((contrast_ne_tokens.size(0), contrast_ne_tokens.size(0)), dtype=torch.bool)
        for aaa, bbb in valid_contrast_pos:
            valid_contrast[aaa, bbb] = True
            valid_contrast[bbb, aaa] = True

        for i in range(contrast_ne_tokens.size(0)):
            valid_contrast[i, i] = False
            positive_contrast[i, i] = False
    else:
        contrast_ne_tokens = None

        valid_contrast = torch.zeros((contrast_prev_output_tokens.size(0), contrast_prev_output_tokens.size(0)), dtype=torch.bool)
        positive_contrast = torch.zeros((contrast_prev_output_tokens.size(0), contrast_prev_output_tokens.size(0)), dtype=torch.bool)

        accumulate_cnt = 0
        for s in samples:
            s_pos = len(s["pos_target"])
            s_neg = len(s["neg_target"])

            valid_contrast[accumulate_cnt:accumulate_cnt + s_pos + s_neg,
                accumulate_cnt:accumulate_cnt + s_pos + s_neg] = True
            positive_contrast[accumulate_cnt:accumulate_cnt + s_pos, accumulate_cnt:accumulate_cnt + s_pos] = True
            accumulate_cnt += s_pos + s_neg

        for i in range(contrast_prev_output_tokens.size(0)):
            valid_contrast[i, i] = False
            positive_contrast[i, i] = False

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }

    batch["contrast_net_input"] = {
        "prev_output_tokens": contrast_prev_output_tokens
    }
    batch["contrast_src_select_index"] = src_select_index
    batch["contrast_ne"] = contrast_ne_tokens
    batch["contrast_target"] = contrast_target
    batch["valid_contrast"] = valid_contrast
    batch["positive_contrast"] = positive_contrast
    batch["ce_pos"] = cross_entropy_pos

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        pos_tgt,
        neg_tgt,
        pos_mapping,
        neg_mapping,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
        max_neg_samples=5,
        cl_seed=0,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.pos_tgt = pos_tgt
        self.neg_tgt = neg_tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id

        self.cl_seed = cl_seed

        # different for each epoch

        rng = np.random.default_rng(seed=cl_seed)
        self.neg_mapping = neg_mapping
        self.neg_shuffle = [rng.permutation(mapping) for mapping in neg_mapping]

        self.pos_mapping = pos_mapping

        self.max_neg_samples = max_neg_samples

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes
            logger.info("bucketing source lengths: {}".format(list(self.src.buckets)))
            if self.tgt is not None:
                self.tgt = BucketPadLengthDataset(
                    self.tgt,
                    sizes=self.tgt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.tgt_sizes = self.tgt.sizes
                logger.info(
                    "bucketing target lengths: {}".format(list(self.tgt.buckets))
                )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

        self.current_epoch = 0

    def get_batch_shapes(self):
        return self.buckets

    def set_epoch(self, epoch):
        self.current_epoch = epoch - 1

    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def __getitem__(self, index):
        src_item = self.src[index]

        pos_tgt_items = [self.pos_tgt[pos_idx] for pos_idx in self.pos_mapping[index]]
        assert pos_tgt_items

        tgt_item = pos_tgt_items[-1].clone()

        pos_ne_items = [torch.ones_like(
            pos_tgt_item,
            dtype=torch.long
        ) for pos_tgt_item in pos_tgt_items]

        for pos_ne_item in pos_ne_items:
            pos_ne_item[-1] = 0

        neg_tgt_indices = self.neg_shuffle[index]

        neg_tgt_items = [
            self.neg_tgt[neg_tgt_indices[i % len(neg_tgt_indices)]]
            for i in range(self.current_epoch * self.max_neg_samples,
                           self.current_epoch * self.max_neg_samples + min(self.max_neg_samples, len(neg_tgt_indices)))
        ]
        neg_ne_items = [torch.ones_like(
            neg_tgt_item,
            dtype=torch.long
        ) for neg_tgt_item in neg_tgt_items]

        for neg_ne_item in neg_ne_items:
            neg_ne_item[-1] = 0

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "neg_target": neg_tgt_items,
            "pos_target": pos_tgt_items,
            "pos_ne": pos_ne_items,
            "neg_ne": neg_ne_items
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = (
                    torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = (
                    torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
                )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
                getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes,
            self.tgt_sizes,
            indices,
            max_sizes,
        )

