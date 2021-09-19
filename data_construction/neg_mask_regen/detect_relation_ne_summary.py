import argparse
from tqdm import tqdm
from spacy.tokens import DocBin

import spacy_stanza

nlp = spacy_stanza.load_pipeline("en", use_gpu=True)


def swap_one(summary):
    summary_doc = nlp(summary)

    return summary_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    with open(args.text_file) as f:
        texts = [line.strip() for line in f]

    summary_doc_bin = DocBin(['LEMMA', 'POS', 'DEP', 'ENT_IOB', 'ENT_TYPE', 'IS_STOP', 'HEAD'])
    for text in tqdm(texts):
        text_doc = swap_one(text)
        summary_doc_bin.add(text_doc)

    with open(args.out_file, 'wb') as f:
        f.write(summary_doc_bin.to_bytes())


if __name__ == '__main__':
    main()