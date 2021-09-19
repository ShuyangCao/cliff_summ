import argparse
import augmentation_ops as ops
from concurrent.futures import ProcessPoolExecutor
import neuralcoref
import regex as re
import spacy
import json
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe

same_ent_swap = ops.SameNERSwap()

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )


def swap_one(document, summary):
    document_doc = nlp(document)
    summary_doc = nlp(summary)

    ret = {}

    summary_tokens = [t.text for t in summary_doc]

    gpt2_summary_tokens = []
    gpt2_bpe_length = []
    bpe_summary_tokens = []
    for token in re.findall(pat, summary):
        gpt2_summary_tokens.append(token.strip())
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
        bpe_summary_tokens.extend(bpe_tokens)
        gpt2_bpe_length.append(len(bpe_tokens))

    ret['summary'] = summary
    ret['tokens'] = summary_tokens
    ret['gpt2_tokens'] = gpt2_summary_tokens
    ret['bpe_tokens'] = bpe_summary_tokens
    ret['bpe_length'] = gpt2_bpe_length

    entity_sample = same_ent_swap.transform({
        "text": document_doc,
        "claim": summary_doc
    })

    replaced_samples = []

    for entity_sample_summary, entity_sample_span, replaced_label in entity_sample:
        entity_sample_summary_tokens = [t.text for t in entity_sample_summary]

        gpt2_entity_summary_tokens = []
        gpt2_entity_bpe_length = []
        bpe_entity_summary_tokens = []
        for token in re.findall(pat, entity_sample_summary.text):
            gpt2_entity_summary_tokens.append(token.strip())
            token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens = [bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")]
            bpe_entity_summary_tokens.extend(bpe_tokens)
            gpt2_entity_bpe_length.append(len(bpe_tokens))

        replaced_samples.append({
            'summary': entity_sample_summary.text,
            'tokens': entity_sample_summary_tokens,
            'gpt2_tokens': gpt2_entity_summary_tokens,
            'bpe_tokens': bpe_entity_summary_tokens,
            'bpe_length': gpt2_entity_bpe_length,
            'swap_span': entity_sample_span,
            'swap_label': replaced_label
        })

    ret['replaced_samples'] = replaced_samples

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('summary')
    parser.add_argument('out_file')
    args = parser.parse_args()

    with open(args.document) as f:
        documents = [line.strip() for line in f]

    with open(args.summary) as f:
        summaries = [line.strip() for line in f]

    with ProcessPoolExecutor() as executor:
        futures = []
        for document, summary in zip(documents, summaries):
            futures.append(executor.submit(swap_one, document, summary))
        results = [future.result() for future in futures]

    with open(args.out_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()