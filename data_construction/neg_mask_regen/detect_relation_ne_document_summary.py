import argparse
from tqdm import tqdm
from spacy.tokens import DocBin

import spacy_stanza

nlp = spacy_stanza.load_pipeline("en", use_gpu=True)


def swap_one(document, summary):
    try:
        source_doc = nlp(document)
    except RecursionError:
        source_doc = nlp(document[:2000])

    summary_doc = nlp(summary)

    return source_doc, summary_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('document')
    parser.add_argument('summary')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    with open(args.document) as f:
        documents = [line.strip() for line in f]

    with open(args.summary) as f:
        summaries = [line.strip() for line in f]

    source_doc_bin = DocBin(['LEMMA', 'POS', 'DEP', 'ENT_IOB', 'ENT_TYPE', 'IS_STOP', 'HEAD'])
    summary_doc_bin = DocBin(['LEMMA', 'POS', 'DEP', 'ENT_IOB', 'ENT_TYPE', 'IS_STOP', 'HEAD'])
    for document, summary in tqdm(zip(documents, summaries)):
        source_doc, summary_doc = swap_one(document, summary)
        source_doc_bin.add(source_doc)
        summary_doc_bin.add(summary_doc)

    with open(args.out_prefix + '.source', 'wb') as f:
        f.write(source_doc_bin.to_bytes())
    with open(args.out_prefix + '.target', 'wb') as f:
        f.write(summary_doc_bin.to_bytes())


if __name__ == '__main__':
    main()