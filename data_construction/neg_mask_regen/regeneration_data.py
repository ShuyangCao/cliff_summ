import argparse
import json
from spacy.training import Alignment
import regex as re
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe


pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )


def create_one(i, source, jsonl):
    x = json.loads(jsonl)
    summary = x['summary']
    ori_tokens = x['tokens']

    gpt2_summary_tokens = []
    gpt2_bpe_length = []
    bpe_summary_tokens = []
    for token in re.findall(pat, summary):
        gpt2_summary_tokens.append(token.strip())
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
        bpe_summary_tokens.extend(bpe_tokens)
        gpt2_bpe_length.append(len(bpe_tokens))

    ori_bpe = bpe_summary_tokens
    ori_gpt2_tokens = gpt2_summary_tokens
    ori_bpe_length = gpt2_bpe_length

    ori_tokens = [x.replace(u'\xa0', ' ') for x in ori_tokens]
    ori_gpt2_tokens = [' ' if x == '' else x for x in ori_gpt2_tokens]

    ori_alignment = Alignment.from_strings(ori_tokens, ori_gpt2_tokens)

    num_valid_ents = len(x['entitys']) + len(x['entity_relations'])
    if num_valid_ents <= 0:
        return []

    bpe_start2meta = defaultdict(list)

    for j, entity in enumerate(x['entitys']):
        entity_start = entity['ent_start']

        bpe_start = sum(ori_bpe_length[:ori_alignment.x2y.dataXd[ori_alignment.x2y.lengths[:entity_start].sum()]])

        bpe_start2meta[bpe_start].append(f'entity_{j}')

    for j, entity_relation in enumerate(x['entity_relations']):
        entity_start = entity_relation['chunk_start']
        for k, head in enumerate(entity_relation['heads']):
            head_start = head['chunk_start']

            if head_start < entity_start:
                bpe_start = sum(
                    ori_bpe_length[:ori_alignment.x2y.dataXd[ori_alignment.x2y.lengths[:head_start].sum()]])
            else:
                bpe_start = sum(
                    ori_bpe_length[:ori_alignment.x2y.dataXd[ori_alignment.x2y.lengths[:entity_start].sum()]])

            bpe_start2meta[bpe_start].append(f'relation_{j}_head{k}')

        for k, head in enumerate(entity_relation['children']):
            head_start = head['chunk_start']

            if head_start < entity_start:
                bpe_start = sum(
                    ori_bpe_length[:ori_alignment.x2y.dataXd[ori_alignment.x2y.lengths[:head_start].sum()]])
            else:
                bpe_start = sum(
                    ori_bpe_length[:ori_alignment.x2y.dataXd[ori_alignment.x2y.lengths[:entity_start].sum()]])

            bpe_start2meta[bpe_start].append(f'relation_{j}_child{k}')

    all_pairs = []
    for bpe_start in bpe_start2meta:
        all_pairs.append((i, source, ori_bpe[:bpe_start], bpe_start, bpe_start2meta[bpe_start]))

    return all_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_file')
    parser.add_argument('jsonl_file')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    with open(args.source_file) as f:
        sources = [line.strip() for line in f]

    with open(args.jsonl_file) as f:
        jsonls = [line.strip() for line in f]

    with ProcessPoolExecutor() as executor:
        futures = []
        for i, (source, jsonl) in enumerate(zip(sources, jsonls)):
            futures.append(executor.submit(create_one, i, source, jsonl))
        results = [future.result() for future in futures]

    with open(args.out_prefix + '.bpe.source', 'w') as fsrc, \
            open(args.out_prefix + '.bpe.target', 'w') as ftgt, \
            open(args.out_prefix + '.other', 'w') as fother:
        for all_pairs in results:
            for i, source, bpe_prefix, bpe_start, meta in all_pairs:
                fsrc.write(source + '\n')
                ftgt.write(' '.join([str(x) for x in bpe_prefix]) + '\n')
                fother.write('{}\t{}\t{}\n'.format(i, bpe_start, ' '.join(meta)))


if __name__ == '__main__':
    main()