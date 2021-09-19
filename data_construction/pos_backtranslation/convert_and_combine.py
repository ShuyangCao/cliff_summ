import argparse
from concurrent.futures import ProcessPoolExecutor
import regex as re
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe

pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )


def select_sentence(aug, ori):
    if aug is not None:
        ms_bpe = []
        for token in re.findall(pat, aug):
            token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
            ms_bpe.extend(bpe_tokens)

        ret = [(ms_bpe, aug)]
    else:
        ret = []

    s_bpe = []
    for token in re.findall(pat, ori):
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
        s_bpe.extend(bpe_tokens)

    ret.append((s_bpe, ori))
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('aug')
    parser.add_argument('ori')
    parser.add_argument('raw_other')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    with open(args.aug) as f:
        augs = [line.strip() for line in f]

    with open(args.ori) as f:
        oris = [line.strip() for line in f]

    with open(args.raw_other) as f:
        raw_others = [int(line.strip()) for line in f]

    aug_map = {ro: aug for ro, aug in zip(raw_others, augs)}

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, ori in enumerate(oris):
            futures.append(executor.submit(select_sentence, aug_map.get(i, None), ori))
        results = [future.result() for future in futures]

    with open(args.out_prefix + '.pos_target', 'w') as ftgt, \
            open(args.out_prefix + '.other', 'w') as fother, \
            open(args.out_prefix + '.combine_target', 'w') as fcomb:
        for i, result in enumerate(results):
            for bpe, raw in result:
                ftgt.write(' '.join([str(x) for x in bpe]) + '\n')
                fcomb.write(raw + '\n')
                fother.write(str(i) + '\n')


if __name__ == '__main__':
    main()
