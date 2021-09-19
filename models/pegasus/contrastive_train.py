from contrastive_data import ContrastiveDataset, DataCollatorForContrastive
from contrastive_model import PegasusForContrastive
from contrastive_trainer import ContrastiveTrainer
from transformers import PegasusTokenizerFast, Seq2SeqTrainingArguments, HfArgumentParser
import os


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    parser.add_argument('--ori_data')
    parser.add_argument('--pos_data')
    parser.add_argument('--neg_data')
    parser.add_argument('--max_neg_samples', type=int)
    parser.add_argument('--max_input_length', type=int)
    parser.add_argument('--max_target_length', type=int)
    training_args, other_args = parser.parse_args_into_dataclasses()

    train_dataset = ContrastiveDataset(
        os.path.join(other_args.ori_data, 'train'),
        os.path.join(other_args.pos_data, 'train'),
        os.path.join(other_args.neg_data, 'train'),
        other_args.max_neg_samples,
        other_args.max_input_length,
        other_args.max_target_length
    )
    valid_dataset = ContrastiveDataset(
        os.path.join(other_args.ori_data, 'valid'),
        os.path.join(other_args.pos_data, 'valid'),
        os.path.join(other_args.neg_data, 'valid'),
        other_args.max_neg_samples,
        other_args.max_input_length,
        other_args.max_target_length
    )

    print('Dataset Loaded.')

    model = PegasusForContrastive.from_pretrained('google/pegasus-large')
    tokenizer = PegasusTokenizerFast.from_pretrained('google/pegasus-large')

    data_collator = DataCollatorForContrastive(tokenizer, model=model)

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == '__main__':
    main()
