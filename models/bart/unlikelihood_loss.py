# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("unlikelihood_loss")
class UnlikelihoodLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        alpha=1.0,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.alpha = alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--alpha', default=1.0, type=float,
                            help='alpha factor')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"], src_select_index=sample["src_select_index"])
        loss, nll_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        if self.sentence_avg:
            sample_size = sample["target"].size(0)
        else:
            sample_size = ntokens
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)  # bsz x seqlen x v
        target = model.get_targets(sample, net_output)  # bsz x seqlen

        negative_indices = sample['negative_indices']  # bsz x seqlen

        pos_lprobs = lprobs[negative_indices == 0]
        pos_target = target[negative_indices == 0]

        neg_lprobs, neg_target, weight = None, None, None
        if negative_indices.eq(1).any():
            neg_lprobs = lprobs[negative_indices == 1]
            neg_target = target[negative_indices == 1]
            neg_lprobs = neg_lprobs.view(-1, lprobs.size(-1))
            neg_target = neg_target.view(-1)

        return pos_lprobs.view(-1, lprobs.size(-1)), pos_target.view(-1), neg_lprobs, neg_target, weight

    def compute_loss(self, model, net_output, sample, reduce=True):
        pos_lprobs, pos_target, neg_lprobs, neg_target, weight = self.get_lprobs_and_target(model, net_output, sample)

        ntokens = pos_target.numel() + (neg_target.numel() if neg_target is not None else 0)

        loss, nll_loss = label_smoothed_nll_loss(
            pos_lprobs,
            pos_target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        if neg_lprobs is not None:
            one_minus_lprobs = torch.log(torch.clamp((1.0 - neg_lprobs.exp()), min=1e-5))
            neg_target = neg_target.unsqueeze(-1)
            neg_loss = -one_minus_lprobs.gather(dim=-1, index=neg_target)
            neg_loss.masked_fill_(neg_target.eq(self.padding_idx), 0.)
            if weight is not None:
                neg_loss = neg_loss * weight

            loss = loss + self.alpha * neg_loss.sum()

        return loss, nll_loss, ntokens

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        have_contrast = utils.item(sum([log.get("n_c", 0) for log in logging_outputs]))
        if have_contrast > 0:
            metrics.log_scalar("n_c", have_contrast)
            contrast_sum = sum(log.get("closs", 0) for log in logging_outputs)

            metrics.log_scalar("closs", contrast_sum, round=3)

            metrics.log_derived(
                "avg_closs",
                lambda meters: safe_round(
                    meters["closs"].sum / meters["n_c"].sum, 3
                )
                if meters["n_c"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
