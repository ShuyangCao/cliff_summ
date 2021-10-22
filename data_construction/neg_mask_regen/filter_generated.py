import os
import argparse
from concurrent.futures import ProcessPoolExecutor
from spacy.tokens import DocBin
from nltk.corpus import wordnet
import spacy
import json
from tqdm import tqdm

wordnet.ensure_loaded()
spacy_nlp = spacy.load('en_core_web_sm', disable=['ner'])


def filter_one(generate_docs, source_doc, target_doc, others):
    # source
    source = source_doc.text
    source_doc_tokens = set([t.text.lower() for t in source_doc])

    spacy_doc = spacy_nlp(source)
    source_token2chunk = {}
    source_stanza_chunks = []
    for noun_chunk in spacy_doc.noun_chunks:
        chunk_char_start = noun_chunk.start_char
        chunk_char_end = noun_chunk.end_char

        chunk = source_doc.char_span(chunk_char_start, chunk_char_end, alignment_mode='expand')
        source_stanza_chunks.append(chunk)

        for j in range(chunk.start, chunk.end):
            source_token2chunk[j] = chunk

    source_entity_relations = []
    for chunk in source_stanza_chunks:
        if not any([tok.ent_iob_ != 'O' for tok in chunk]):
            continue

        entity_relation = {
            'chunk': chunk.text,
            'chunk_start': chunk.start,
            'chunk_end': chunk.end,
        }

        heads = []
        children = []
        for tok in chunk:
            if (
                    tok.head.i >= chunk.end or tok.head.i < chunk.start) and (
                    not tok.head.is_stop or tok.head.text == 'US') and not tok.head.is_punct:
                start = tok.head.i

                if start in source_token2chunk:
                    end = source_token2chunk[start].end
                    start = source_token2chunk[start].start
                else:
                    end = tok.head.i + 1
                # print(tok.head.text, '----', summary_doc[start:end].text, '----', summary)
                heads.append({
                    'head_i': tok.head.i,
                    'head': tok.head.text,
                    'chunk': source_doc[start:end].text,
                    'chunk_start': start,
                    'chunk_end': end,
                    'dep': tok.dep_
                })
            for child in tok.children:
                if (child.i >= chunk.end or child.i < chunk.start) and (
                        not tok.head.is_stop or tok.head.text == 'US') and not child.is_punct:
                    start = child.i

                    if start in source_token2chunk:
                        end = source_token2chunk[start].end
                        start = source_token2chunk[start].start
                    else:
                        end = child.i + 1

                    children.append({
                        'child_i': child.i,
                        'child': child.text,
                        'chunk': source_doc[start:end].text,
                        'chunk_start': start,
                        'chunk_end': end,
                        'dep': child.dep_
                    })

        entity_relation['heads'] = heads
        entity_relation['children'] = children

        source_entity_relations.append(entity_relation)

    # target
    target = target_doc.text
    target_doc_tokens = set([t.text.lower() for t in target_doc])

    spacy_doc = spacy_nlp(target)
    target_token2chunk = {}
    target_stanza_chunks = []
    for noun_chunk in spacy_doc.noun_chunks:
        chunk_char_start = noun_chunk.start_char
        chunk_char_end = noun_chunk.end_char

        chunk = target_doc.char_span(chunk_char_start, chunk_char_end, alignment_mode='expand')
        target_stanza_chunks.append(chunk)

        for j in range(chunk.start, chunk.end):
            target_token2chunk[j] = chunk

    target_entity_relations = []
    for chunk in target_stanza_chunks:
        if not any([tok.ent_iob_ != 'O' for tok in chunk]):
            continue

        entity_relation = {
            'chunk': chunk.text,
            'chunk_start': chunk.start,
            'chunk_end': chunk.end,
        }

        heads = []
        children = []
        for tok in chunk:
            if (
                    tok.head.i >= chunk.end or tok.head.i < chunk.start) and (
                    not tok.head.is_stop or tok.head.text == 'US') and not tok.head.is_punct:
                start = tok.head.i

                if start in target_token2chunk:
                    end = target_token2chunk[start].end
                    start = target_token2chunk[start].start
                else:
                    end = tok.head.i + 1
                # print(tok.head.text, '----', summary_doc[start:end].text, '----', summary)
                heads.append({
                    'head_i': tok.head.i,
                    'head': tok.head.text,
                    'chunk': target_doc[start:end].text,
                    'chunk_start': start,
                    'chunk_end': end,
                    'dep': tok.dep_
                })
            for child in tok.children:
                if (child.i >= chunk.end or child.i < chunk.start) and (
                        not tok.head.is_stop or tok.head.text == 'US') and not child.is_punct:
                    start = child.i

                    if start in target_token2chunk:
                        end = target_token2chunk[start].end
                        start = target_token2chunk[start].start
                    else:
                        end = child.i + 1

                    children.append({
                        'child_i': child.i,
                        'child': child.text,
                        'chunk': target_doc[start:end].text,
                        'chunk_start': start,
                        'chunk_end': end,
                        'dep': child.dep_
                    })

        entity_relation['heads'] = heads
        entity_relation['children'] = children

        target_entity_relations.append(entity_relation)

    all_valid_entity_summaries = []
    all_valid_relation_summaries = []

    seen_ent = set()
    seen_relation = set()

    prev_end = 0
    for other in others:
        other_end = prev_end + other[2]

        for new_summary_doc in generate_docs[prev_end:other_end]:
            new_summary = new_summary_doc.text

            if new_summary == target:
                continue

            if 'entity' in other[1]:
                if new_summary not in seen_ent:
                    seen_ent.add(new_summary)
                    new_entities = []
                    for ent in new_summary_doc.ents:
                        ent_text = ent.text
                        if ent.label_ in ['TIME', 'DATE']:
                            if ent_text.endswith("'s"):
                                ent_text = ent_text[:-2]
                            if ent_text.lower() in source.lower():
                                continue
                            if ent_text.lower() in target.lower():
                                continue
                        elif ent.label_ == 'PERSON':
                            if not any([t.text.lower() not in source_doc_tokens for t in ent if
                                        t.text != "'s" and (not t.is_stop or t.text == 'US')]):
                                continue
                            if not any([t.text.lower() not in target_doc_tokens for t in ent if
                                        t.text != "'s" and (not t.is_stop or t.text == 'US')]):
                                continue
                        else:
                            if not all([t.text.lower() not in source_doc_tokens for t in ent if
                                        t.text != "'s" and (not t.is_stop or t.text == 'US')]):
                                continue
                            if not all([t.text.lower() not in target_doc_tokens for t in ent if
                                        t.text != "'s" and (not t.is_stop or t.text == 'US')]):
                                continue
                        new_entities.append((ent.text, ent.start, ent.end, ent.label_))
                    if new_entities:
                        all_valid_entity_summaries.append({
                            'summary': new_summary,
                            'new_entities': new_entities
                        })

            if 'relation' in other[1]:
                if new_summary not in seen_relation:
                    seen_relation.add(new_summary)
                    new_spacy_doc = spacy_nlp(new_summary)

                    token2chunk = {}
                    stanza_chunks = []
                    for noun_chunk in new_spacy_doc.noun_chunks:
                        chunk_char_start = noun_chunk.start_char
                        chunk_char_end = noun_chunk.end_char

                        chunk = new_summary_doc.char_span(chunk_char_start, chunk_char_end, alignment_mode='expand')
                        stanza_chunks.append(chunk)

                        for k in range(chunk.start, chunk.end):
                            token2chunk[k] = chunk

                    new_relations = []

                    for chunk in stanza_chunks:
                        # relation in the generated summaries
                        if not any([tok.ent_iob_ != 'O' for tok in chunk]):
                            continue

                        chunk_tokens = set([tok.lemma_.lower() for tok in chunk if (not tok.is_stop or tok.text == 'US') and not tok.is_punct])

                        for tok in chunk:
                            if (
                                    tok.head.i >= chunk.end or tok.head.i < chunk.start) and (not tok.head.is_stop or tok.head.text == 'US') and not tok.head.is_punct:
                                start = tok.head.i
                                # new head relation?

                                if start in token2chunk:
                                    end = token2chunk[start].end
                                    start = token2chunk[start].start
                                else:
                                    end = tok.head.i + 1

                                head_chunk_tokens = set([lm for head_tok in new_summary_doc[start:end] for ss in wordnet.synsets(head_tok.lemma_.lower()) for lm in
                                     ss.lemma_names()])
                                head_dep = tok.dep_

                                # whether the relation occurs in ref
                                for ref_entity_relation in target_entity_relations:
                                    if any([ref_tok.lemma_.lower() in chunk_tokens for ref_tok in target_doc[ref_entity_relation['chunk_start']: ref_entity_relation['chunk_end']]]) and \
                                            any([ref_head['dep'] == head_dep and any([ref_tok.lemma_.lower() in head_chunk_tokens for ref_tok in target_doc[ref_head['chunk_start']:ref_head['chunk_end']]])
                                            for ref_head in ref_entity_relation['heads']]):
                                        break  # find match in reference
                                else:
                                    for ref_entity_relation in source_entity_relations:
                                        if any([ref_tok.lemma_.lower() in chunk_tokens for ref_tok in source_doc[
                                                                                                      ref_entity_relation[
                                                                                                          'chunk_start']:
                                                                                                      ref_entity_relation[
                                                                                                          'chunk_end']]]) and \
                                                any([ref_head['dep'] == head_dep and any(
                                                    [ref_tok.lemma_.lower() in head_chunk_tokens for ref_tok in
                                                     source_doc[ref_head['chunk_start']:ref_head['chunk_end']]])
                                                     for ref_head in ref_entity_relation['heads']]):
                                            break  # find match in source
                                    else:
                                        new_relations.append((chunk.text, chunk.start, chunk.end, new_summary_doc[start:end].text, start, end, tok.dep_))

                            for child in tok.children:
                                if (
                                        child.i >= chunk.end or child.i < chunk.start) and (not child.is_stop or child.text == 'US') and not child.is_punct:
                                    start = child.i
                                    # new child relation?
                                    if start in token2chunk:
                                        end = token2chunk[start].end
                                        start = token2chunk[start].start
                                    else:
                                        end = child.i + 1

                                    head_chunk_tokens = set([lm for head_tok in new_summary_doc[start:end] for ss in
                                                             wordnet.synsets(head_tok.lemma_.lower()) for lm in
                                                             ss.lemma_names()])
                                    head_dep = child.dep_

                                    for ref_entity_relation in target_entity_relations:
                                        if any([ref_tok.lemma_.lower() in chunk_tokens for ref_tok in target_doc[
                                                                                                      ref_entity_relation[
                                                                                                          'chunk_start']:
                                                                                                      ref_entity_relation[
                                                                                                          'chunk_end']]]) and \
                                                any([ref_head['dep'] == head_dep and any(
                                                    [ref_tok.lemma_.lower() in head_chunk_tokens for ref_tok in
                                                     target_doc[ref_head['chunk_start']:ref_head['chunk_end']]])
                                                     for ref_head in ref_entity_relation['children']]):
                                            break  # find in ref
                                    else:
                                        for ref_entity_relation in source_entity_relations:
                                            if any([ref_tok.lemma_.lower() in chunk_tokens for ref_tok in source_doc[
                                                                                                          ref_entity_relation[
                                                                                                              'chunk_start']:
                                                                                                          ref_entity_relation[
                                                                                                              'chunk_end']]]) and \
                                                    any([ref_head['dep'] == head_dep and any(
                                                        [ref_tok.lemma_.lower() in head_chunk_tokens for ref_tok in
                                                         source_doc[ref_head['chunk_start']:ref_head['chunk_end']]])
                                                         for ref_head in ref_entity_relation['children']]):
                                                break  # find in source
                                        else:
                                            new_relations.append((chunk.text, chunk.start, chunk.end,
                                                                  new_summary_doc[start:end].text, start, end, child.dep_))

                    if new_relations:
                        all_valid_relation_summaries.append({
                            'summary': new_summary,
                            'new_relations': new_relations
                        })

        prev_end = other_end

    return {
        'id': others[0][0],
        'ori_summary': target,
        'new_entity_summaries': all_valid_entity_summaries,
        'new_relation_summarys': all_valid_relation_summaries
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-docbins', nargs='+')
    parser.add_argument('--source-docbins', nargs='+')
    parser.add_argument('--target-docbins', nargs='+')
    parser.add_argument('--other')
    parser.add_argument('output')
    args = parser.parse_args()

    nlp = spacy.blank("en")

    generate_docs = []
    for generated_docbin in args.generated_docbins:
        with open(generated_docbin, 'rb') as f:
            byte_data = f.read()
            doc_bin = DocBin().from_bytes(byte_data)
            generate_docs.extend(list(doc_bin.get_docs(nlp.vocab)))

    others = []
    with open(args.other) as f:
        for line in f:
            if '\t' in line:
                id, _, gen_type, num = line.strip().split('\t')
            else:
                id, _, gen_type, num = line.strip().split(' ')
            id = int(id)
            num = int(num)
            others.append((id, gen_type, num))

    print(len(generate_docs))

    assert sum([x[2] for x in others]) == len(generate_docs)

    source_docs = []
    for source_docbin in args.source_docbins:
        with open(source_docbin, 'rb') as f:
            byte_data = f.read()
            doc_bin = DocBin().from_bytes(byte_data)
            source_docs.extend(list(doc_bin.get_docs(nlp.vocab)))

    print(len(source_docs))

    target_docs = []
    for target_docbin in args.target_docbins:
        with open(target_docbin, 'rb') as f:
            byte_data = f.read()
            doc_bin = DocBin().from_bytes(byte_data)
            target_docs.extend(list(doc_bin.get_docs(nlp.vocab)))

    print(len(target_docs))

    with ProcessPoolExecutor() as executor:
        futures = []

        other_batch = [others[0]]
        start = 0
        end = others[0][2]
        for other in tqdm(others[1:]):
            if other[0] != other_batch[0][0]:
                futures.append(filter_one(generate_docs[start:end], source_docs[other_batch[0][0]], target_docs[other_batch[0][0]], other_batch))
                other_batch = [other]
                start = end
                end = end + other[2]
            else:
                other_batch.append(other)
                end = end + other[2]
        if other_batch:
            futures.append(
                filter_one(generate_docs[start:end], source_docs[other_batch[0][0]], target_docs[other_batch[0][0]],
                           other_batch))

        results = futures

    results = {result['id']: result for result in results}

    with open(args.output, 'w') as f:
        for i in range(len(target_docs)):
            f.write(json.dumps(results.get(i, {})) + '\n')


if __name__ == '__main__':
    main()