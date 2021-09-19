from transformers import PegasusTokenizerFast, PegasusForConditionalGeneration
import torch

import argparse
import os
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('model_path')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    with open(args.source) as f:
        sources = [line.strip() for line in f]

    tokenizer = PegasusTokenizerFast.from_pretrained(args.model_path)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_path)
    model.cuda()
    model.eval()

    print('Model loaded')

    all_outputs = []

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(sources), 8)):
            batch_sources = sources[batch_start:batch_start+8]
            inputs = tokenizer(batch_sources, return_tensors="pt", truncation=True, max_length=1024, padding=True).to('cuda')
            outputs = model.generate(**inputs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_outputs.extend(decoded)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'formatted-test.txt'), 'w') as f:
        for output in all_outputs:
            f.write(output + '\n')


if __name__ == '__main__':
    main()
