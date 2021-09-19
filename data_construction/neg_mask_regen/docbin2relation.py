from spacy.tokens import DocBin
import argparse
import spacy
import spacy_stanza
import json
from concurrent.futures import ProcessPoolExecutor

spacy_nlp = spacy.load('en_core_web_sm', disable=['ner'])


def doc2relation(summary_doc):
    ret = {}

    summary = summary_doc.text
    summary_tokens = [t.text for t in summary_doc]
    summary_tokens_ws = [t.text_with_ws for t in summary_doc]

    spacy_doc = spacy_nlp(summary)

    entitys = []
    entity_relations = []
    for ent in summary_doc.ents:
        entity = {
            'entity': ent.text,
            'ent_start': ent.start,
            'ent_end': ent.end,
            'ent_start_char': ent.start_char,
            'ent_end_char': ent.end_char,
            'ent_type': ent.label_
        }
        entitys.append(entity)

    token2chunk = {}
    stanza_chunks = []
    for noun_chunk in spacy_doc.noun_chunks:
        chunk_char_start = noun_chunk.start_char
        chunk_char_end = noun_chunk.end_char

        chunk = summary_doc.char_span(chunk_char_start, chunk_char_end, alignment_mode='expand')
        stanza_chunks.append(chunk)

        for j in range(chunk.start, chunk.end):
            token2chunk[j] = chunk

    for chunk in stanza_chunks:
        if not any([tok.ent_iob_ != 'O' for tok in chunk]):
            continue

        chunk_lemma_tokens = list(set(
            [tok.lemma_ for tok in summary_doc[chunk.start:chunk.end] if
             (not tok.is_stop or tok.text == 'US') and not tok.is_punct]))

        entity_relation = {
            'chunk': chunk.text,
            'chunk_start': chunk.start,
            'chunk_end': chunk.end,
            'chunk_lemma_tokens': chunk_lemma_tokens
        }

        heads = []
        children = []
        for tok in chunk:
            if (tok.head.i >= chunk.end or tok.head.i < chunk.start) and (
                    not tok.head.is_stop or tok.head.text == 'US') and not tok.head.is_punct:
                start = tok.head.i

                if start in token2chunk:
                    end = token2chunk[start].end
                    start = token2chunk[start].start
                else:
                    end = tok.head.i + 1

                chunk_lemma_tokens = list(set(
                    [tok.lemma_ for tok in summary_doc[start:end] if
                     (not tok.is_stop or tok.text == 'US') and not tok.is_punct]))

                heads.append({
                    'head_i': tok.head.i,
                    'head': tok.head.text,
                    'chunk': summary_doc[start:end].text,
                    'chunk_start': start,
                    'chunk_end': end,
                    'chunk_lemma_tokens': chunk_lemma_tokens,
                    'dep': tok.dep_
                })
            for child in tok.children:
                if (child.i >= chunk.end or child.i < chunk.start) and (
                        not child.is_stop or child.text == 'US') and not child.is_punct:
                    start = child.i

                    if start in token2chunk:
                        end = token2chunk[start].end
                        start = token2chunk[start].start
                    else:
                        end = child.i + 1

                    chunk_lemma_tokens = list(set(
                        [tok.lemma_ for tok in summary_doc[start:end] if
                         (not tok.is_stop or tok.text == 'US') and not tok.is_punct]))

                    children.append({
                        'child_i': child.i,
                        'child': child.text,
                        'chunk': summary_doc[start:end].text,
                        'chunk_start': start,
                        'chunk_end': end,
                        'chunk_lemma_tokens': chunk_lemma_tokens,
                        'dep': child.dep_
                    })

        entity_relation['heads'] = heads
        entity_relation['children'] = children

        entity_relations.append(entity_relation)

    ret['summary'] = summary
    ret['tokens'] = summary_tokens
    ret['tokens_ws'] = summary_tokens_ws
    ret['entitys'] = entitys
    ret['entity_relations'] = entity_relations

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_bin')
    parser.add_argument('out_jsonl')
    args = parser.parse_args()

    nlp = spacy_stanza.load_pipeline("en", use_gpu=False)

    with open(args.doc_bin, 'rb') as f:
        byte_data = f.read()
        doc_bin = DocBin().from_bytes(byte_data)
        docs = list(doc_bin.get_docs(nlp.vocab))

    with ProcessPoolExecutor() as executor:
        futures = []
        for doc in docs:
            futures.append(executor.submit(doc2relation, doc))
        results = [future.result() for future in futures]

    with open(args.out_jsonl, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()