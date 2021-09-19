import argparse
import os
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-dir', nargs='+')
    args = parser.parse_args()

    bpe = GPT2BPE(GPT2BPEConfig()).bpe

    for sp in ['train', 'valid', 'test']:
        for generate_dir in args.generate_dir:

            all_samples = []
            sample = []
            all_bpe_samples = []
            bpe_sample = []
            if not os.path.exists(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))):
                continue
            with open(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))) as f:
                for line in f:
                    if line[0] == 'S':
                        if sample:
                            all_samples.append((sample_id, sample))
                            all_bpe_samples.append((sample_id, bpe_sample))
                            sample = []
                            bpe_sample = []
                        try:
                            sample_id, sent = line.strip().split('\t')
                        except:
                            sample_id = line.strip()
                            sent = '2333'
                        sample_id = sample_id.split('-')[1]
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', 'madeupword0000', 'madeupword0001', 'madeupword0002'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'T':
                        try:
                            sent = line.strip().split('\t')[1]
                        except:
                            sent = '2333'
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', 'madeupword0000', 'madeupword0001', 'madeupword0002'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'H':
                        sent = line.strip().split('\t')[-1]
                        bpe_sample.append(sent)
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', 'madeupword0000', 'madeupword0001', 'madeupword0002'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.strip()
                        sample.append(sent)
            if sample:
                all_samples.append((sample_id, sample))
                all_bpe_samples.append((sample_id, bpe_sample))
                sample = []

            all_samples = sorted(all_samples, key=lambda x: int(x[0]))
            all_samples = [x[1] for x in all_samples]
            all_bpe_samples = sorted(all_bpe_samples, key=lambda x: int(x[0]))
            all_bpe_samples = [x[1] for x in all_bpe_samples]

            with open(os.path.join(generate_dir, 'formatted-{}.txt'.format(sp)), 'w') as f:
                for sample in all_samples:
                    for candidate in sample[2:]:
                        f.write(candidate + '\n')


if __name__ == '__main__':
    main()
