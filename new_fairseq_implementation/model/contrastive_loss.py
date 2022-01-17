# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging.meters import safe_round
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class ContrastiveLossCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    alpha: float = field(
        default=1.0
    )
    tau: float = field(
        default=1.0
    )


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


@register_criterion("contrastive_loss", dataclass=ContrastiveLossCriterionConfig)
class ContrastiveLossCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        alpha=1.0,
        tau=1.0
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.alpha = alpha
        self.tau = tau

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        ce_net_output, contrast_net_output = model(**sample["net_input"], **sample["contrast_net_input"],
                                                   classification_head_name="contrast",
                                                   src_select_index=sample["contrast_src_select_index"])

        ce_pos = sample["ce_pos"]
        ce_net_output = ce_net_output[ce_pos]

        loss, nll_loss = self.compute_loss(model, ce_net_output, sample, reduce=reduce)

        loss = loss / sample["ntokens"]

        contrast_loss = self.compute_contrast_loss(model, contrast_net_output, sample)
        loss += self.alpha * contrast_loss

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if contrast_loss is not None:
            logging_output["closs"] = utils.item(contrast_loss.data)
            logging_output["n_c"] = 1
        else:
            logging_output["closs"] = 0
            logging_output["n_c"] = 0
        return loss, sample_size, logging_output

    def compute_contrast_loss(self, model, contrast_net_output, sample):
        contrast_ne = sample["contrast_ne"]  # B x T
        if contrast_ne is not None:
            ne_representation = contrast_net_output.masked_fill((contrast_ne == 0).unsqueeze(-1), 0)  # B x T x C
            representation = ne_representation.sum(dim=1)
            representation_ne_denom = contrast_ne.sum(dim=1, keepdim=True)
            representation = representation / torch.max(representation_ne_denom, 1e-8 * torch.ones_like(representation_ne_denom))
        else:
            representation = contrast_net_output[:, -1]

        representation_n = representation.norm(dim=-1, keepdim=True)
        representation_norm = representation / torch.max(representation_n, 1e-8 * torch.ones_like(representation_n))

        similarity = torch.matmul(representation_norm, representation_norm.transpose(0, 1))  # pos+neg x pos+neg
        similarity = similarity / self.tau
        similarity = similarity.exp()
        similarity = similarity.masked_fill(~sample["valid_contrast"], 0.)
        denominator = similarity.sum(dim=-1, keepdim=True)  # pos+neg

        denom_similarity = similarity / torch.max(denominator, 1e-8 * torch.ones_like(denominator))  # pos+neg x pos+neg

        loss = denom_similarity[sample["positive_contrast"]]

        loss = - loss.log()

        loss_denom = sample["positive_contrast"].sum()

        loss = loss.sum() / torch.max(loss_denom, 1e-8 * torch.ones_like(loss_denom))
        return loss


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = sample["contrast_target"]
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        net_output = (net_output, None)
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

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
