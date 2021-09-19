import os
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_jsonl')
    parser.add_argument('out_prefix')
    parser.add_argument('--threshold', type=float, default=0.21)
    args = parser.parse_args()

    with open(args.candidate_jsonl) as f, open(args.out_prefix + '.neg_target', 'w') as ftgt, \
            open(args.out_prefix + '.raw_target', 'w') as frawtgt, open(args.out_prefix + '.other', 'w') as fother:
        for i, line in enumerate(f):
            x = json.loads(line)
            for sent, bpe_sent, _, _, min_prob in x['candidates']:
                if min_prob < args.threshold:
                    ftgt.write(bpe_sent + '\n')
                    frawtgt.write(sent + '\n')
                    fother.write(str(i) + '\n')


if __name__ == '__main__':
    main()
