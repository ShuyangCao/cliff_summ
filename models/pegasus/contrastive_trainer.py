from transformers.training_args import TrainingArguments
from dataclasses import dataclass, field


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from transformers import PreTrainedModel, Trainer, logging

logger = logging.get_logger(__name__)



class ContrastiveTrainer(Trainer):
    def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers
        )


    def _compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        ce_pos = inputs.pop("ce_pos")
        positive_contrast = inputs.pop("positive_contrast")
        valid_contrast = inputs.pop("valid_contrast")

        model_output = model(**inputs, use_cache=False)
        logits = model_output.logits

        ce_logits = logits[ce_pos]
        ce_targets = labels[ce_pos]

        loss = self.label_smoother((ce_logits,), ce_targets)

        representation = model_output.contrast_states  # bsz x seqlen x dim
        ne_representation = representation.masked_fill((labels == -100).unsqueeze(-1), 0)  # B x T x C
        representation = ne_representation.sum(dim=1)
        representation_ne_denom = (labels != -100).sum(dim=1, keepdim=True)
        representation = representation / torch.max(representation_ne_denom,
                                                    1e-8 * torch.ones_like(representation_ne_denom))

        representation_n = representation.norm(dim=-1, keepdim=True)
        representation_norm = representation / torch.max(representation_n, 1e-8 * torch.ones_like(representation_n))
        similarity = torch.matmul(representation_norm, representation_norm.transpose(0, 1))  # pos+neg x pos+neg
        similarity = similarity.exp()
        similarity = similarity.masked_fill(~valid_contrast, 0.)
        denominator = similarity.sum(dim=-1, keepdim=True)  # pos+neg

        denom_similarity = similarity / torch.max(denominator, 1e-8 * torch.ones_like(denominator))  # pos+neg x pos+neg
        contrast_loss = denom_similarity[positive_contrast]
        contrast_loss = - contrast_loss.log()
        contrast_loss_denom = positive_contrast.sum()

        contrast_loss = contrast_loss.sum() / torch.max(contrast_loss_denom,
                                                        1e-8 * torch.ones_like(contrast_loss_denom))

        return loss + contrast_loss, logits

    def compute_loss(self, model, inputs):
        loss, _ = self._compute_loss(model, inputs)
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # compute loss on predict data
            loss, _ = self._compute_loss(model, inputs)

        loss = loss.mean().detach()
        return (loss, None, None)


    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor