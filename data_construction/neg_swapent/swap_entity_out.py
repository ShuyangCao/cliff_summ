import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file')
    parser.add_argument('swap_jsonl')
    parser.add_argument('swap_out_prefix')
    args = parser.parse_args()

    with open(args.source_file) as f:
        sources = [line.strip() for line in f]

    swap_pairs = []
    with open(args.swap_jsonl) as f:
        for i, line in enumerate(f):
            x = json.loads(line)
            ori_text = x['summary']
            ori_bpe = x['bpe_tokens']

            swap_pairs.append((i, sources[i], ori_bpe, ori_text, -1))

            for sample in x['replaced_samples']:
                swp_text = sample['summary']
                swp_bpe = sample['bpe_tokens']

                swap_pairs.append((i, sources[i], swp_bpe, swp_text, 0))

    with open(args.swap_out_prefix + '.neg_target', 'w') as ftgt, \
            open(args.swap_out_prefix + '.raw_target', 'w') as frawtgt, \
            open(args.swap_out_prefix + '.other', 'w') as fother:
        for i, source, ori_bpe, swp_text, bpe_start in swap_pairs:
            ftgt.write(' '.join([str(x) for x in ori_bpe]) + '\n')
            frawtgt.write(swp_text + '\n')
            fother.write('{}\t{}\n'.format(i, bpe_start))


if __name__ == '__main__':
    main()
