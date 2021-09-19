import argparse
from concurrent.futures import ProcessPoolExecutor
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
import random

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe


def add_prefix(others, generated_texts, prefixs, beam_size):
    out_texts = []
    for i, (other, prefix) in enumerate(zip(others, prefixs)):
        beam = generated_texts[i * beam_size:(i + 1) * beam_size]
        prefix_text = bpe_encoder.decode([int(token) for token in prefix.split()]).strip()
        if prefix_text:
            prefix_text = prefix_text + ' '
        beam = [prefix_text + b.strip() for b in beam[:beam_size]]
        out_texts.append((beam, other + [str(len(beam))]))
    if len(out_texts) > 4:
        out_texts = random.sample(out_texts, 4)
    return out_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix_file')
    parser.add_argument('generated_file')
    parser.add_argument('other_file')
    parser.add_argument('out_file')
    parser.add_argument('new_other_file')
    parser.add_argument('--cnndm', action='store_true')
    args = parser.parse_args()

    if args.cnndm:
        beam_size = 3
    else:
        beam_size = 10

    generated_texts = []
    with open(args.generated_file) as f:
        for line in f:
            generated_texts.append(line.strip())

    others = []
    with open(args.other_file) as f:
        for line in f:
            others.append(line.strip().split('\t'))

    prefixs = []
    with open(args.prefix_file) as f:
        for line in f:
            prefixs.append(line)

    current_batch = [others[0]]
    j = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        for other in others[1:]:
            if other[0] != current_batch[0][0]:
                futures.append(executor.submit(add_prefix, current_batch, generated_texts[j * beam_size:j * beam_size + len(current_batch) * beam_size], prefixs[j:j + len(current_batch)], beam_size))
                j += len(current_batch)
                current_batch = [other]
            else:
                current_batch.append(other)
        if current_batch:
            futures.append(
                executor.submit(add_prefix, current_batch, generated_texts[j * beam_size:j + len(current_batch) * beam_size],
                                prefixs[j:j + len(current_batch)], beam_size))
        results = [future.result() for future in futures]

    with open(args.out_file, 'w') as f, open(args.new_other_file, 'w') as fother:
        for result in results:
            for beam, new_other in result:
                for output in beam:
                    f.write(output + '\n')
                fother.write('\t'.join(new_other) + '\n')


if __name__ == '__main__':
    main()