import argparse
import math
from collections import defaultdict
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
import regex as re
from spacy.training import Alignment
import spacy
from concurrent.futures import ProcessPoolExecutor
import json


nlp = spacy.load('en_core_web_sm')

pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )


bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe


def find_consecutive_pos(poses):
    current_pos = None
    current_idx = None
    new_poses = []
    map_first = []
    for i, pos in enumerate(poses):
        pos = pos if pos in ['PROPN', 'NUM'] else 'OTHER'
        if pos == current_pos:
            new_poses.append('OTHER')
            map_first.append(current_idx)
        else:
            new_poses.append(pos)
            current_pos = pos
            current_idx = i
            map_first.append(current_idx)
    return new_poses, map_first


def check_one(sents, bpe_sents, accumulate_probs, bpe_probs):
    candidates = []
    for sent, bpe_sent, accumulate_prob, bpe_prob in zip(sents, bpe_sents, accumulate_probs, bpe_probs):
        sent_doc = nlp(sent)
        spacy_tokens = [t.text for t in sent_doc]

        gpt2_summary_tokens = []
        gpt2_bpe_length = []
        bpe_summary_tokens = []
        for token in re.findall(pat, sent):
            gpt2_summary_tokens.append(token.strip())
            token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
            bpe_summary_tokens.extend(bpe_tokens)
            gpt2_bpe_length.append(len(bpe_tokens))

        if ' '.join([str(x) for x in bpe_summary_tokens]) != bpe_sent:
            continue
        if len(bpe_prob) != len(bpe_summary_tokens):
            continue

        spacy_tokens = [x.replace(u'\xa0', ' ') for x in spacy_tokens]
        gpt2_summary_tokens = [' ' if x == '' else x for x in gpt2_summary_tokens]

        bpe_summary_pos = [None for _ in bpe_summary_tokens]
        bpe_alignment = Alignment.from_strings(spacy_tokens, gpt2_summary_tokens)
        entity_bpe = [0 for _ in bpe_summary_tokens]
        try:
            for i, _ in enumerate(spacy_tokens):
                bpe_start = sum(
                    gpt2_bpe_length[:(bpe_alignment.x2y.dataXd[bpe_alignment.x2y.lengths[:i].sum()] if
                                       bpe_alignment.x2y.lengths[:i].sum() < len(
                                           gpt2_bpe_length) else len(gpt2_bpe_length))])
                bpe_end = sum(
                    gpt2_bpe_length[:(bpe_alignment.x2y.dataXd[bpe_alignment.x2y.lengths[:i+1].sum()] if
                                      bpe_alignment.x2y.lengths[:i+1].sum() < len(
                                          gpt2_bpe_length) else len(gpt2_bpe_length))])
                for j in range(bpe_start, bpe_end):
                    bpe_summary_pos[j] = sent_doc[i].pos_
        except IndexError:
            continue

        for ent in sent_doc.ents:
            ent_start = ent.start
            ent_end = ent.end

            ent_bpe_start = sum(
                gpt2_bpe_length[:bpe_alignment.x2y.dataXd[bpe_alignment.x2y.lengths[:ent_start].sum()]])
            ent_bpe_end = sum(gpt2_bpe_length[:(bpe_alignment.x2y.dataXd[bpe_alignment.x2y.lengths[:ent_end].sum()] if
                                              bpe_alignment.x2y.lengths[:ent_end].sum() < len(
                                                  gpt2_bpe_length) else len(gpt2_bpe_length))])

            for j in range(ent_bpe_start, ent_bpe_end):
                entity_bpe[j] = 1

        assert len(bpe_summary_pos) == len(bpe_prob)

        first_pos = find_consecutive_pos(bpe_summary_pos)[0]

        propn_num_probs = [x for x, pos in zip(bpe_prob, first_pos) if
                           pos in ['PROPN', 'NUM']]
        min_prob = min(propn_num_probs) if propn_num_probs else 1.

        candidates.append((sent, bpe_sent, ' '.join([str(x) for x in entity_bpe]), accumulate_prob, min_prob))

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('generate_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    beam_outs = defaultdict(list)
    bpe_outs = defaultdict(list)
    accumulate_probs = defaultdict(list)
    token_probs = defaultdict(list)
    with open(args.generate_file) as f:
        for line in f:
            if line[:1] == 'H':
                id_part, probability, bpe_sent = line.strip().split('\t')
                sample_id = int(id_part[2:])
                sent = bpe_encoder.decode([
                    int(tok) if tok not in {'<unk>', '<mask>'} else tok
                    for tok in bpe_sent.split()
                ])
                beam_outs[sample_id].append(sent)
                bpe_outs[sample_id].append(bpe_sent)
                accumulate_probs[sample_id].append(math.pow(2, float(probability)))

            if line[:1] == 'P':
                id_part, probability_entropy = line.strip().split('\t')
                sample_id = int(id_part[2:])
                bpe_probs = [math.pow(2, float(x)) for x in probability_entropy.split()[:-1]]
                token_probs[sample_id].append(bpe_probs)

    with ProcessPoolExecutor() as executor:
        futures = []
        for id in sorted(beam_outs):
            futures.append(executor.submit(check_one, beam_outs[id], bpe_outs[id], accumulate_probs[id], token_probs[id]))
        results = [future.result() for future in futures]

    with open(args.out_file, 'w') as f:
        for result in results:
            f.write(json.dumps({'candidates': result}) + '\n')


if __name__ == '__main__':
    main()