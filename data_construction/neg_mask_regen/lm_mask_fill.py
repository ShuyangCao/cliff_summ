import argparse
import json
from fairseq.models.bart import BARTModel
from tqdm import tqdm
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('relation_jsonl')
    parser.add_argument('out')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()

    bart = BARTModel.from_pretrained('/data2/shuyang/pretrain_language_models/bart.large', checkpoint_file='model.pt')
    bart.eval()
    bart.cuda()

    with open(args.relation_jsonl) as f:
        lines = f.readlines()

    if args.start is not None and args.end is not None:
        lines = lines[args.start:args.end]

    new_summaries = []
    for line in tqdm(lines):
        x = json.loads(line)

        summary = x['summary']
        summary_token_ws = x['tokens_ws']
        if len(x['entity_relations'] + x['entitys']) == 0:
            new_summaries.append({'summary': summary, 'new_summary': []})
            continue

        # doc = nlp(summary)
        inputs = []
        for entity in x['entitys']:
            entity_start = entity['ent_start']
            entity_end = entity['ent_end']

            if summary_token_ws[entity_end - 1][-1] == ' ':
                entity_mask = '<mask> '
            else:
                entity_mask = '<mask>'

            inp = []
            inp.extend(summary_token_ws[:entity_start])
            inp.append(entity_mask)
            inp.extend(summary_token_ws[entity_end:])
            inputs.append(''.join(inp))

        num_entity = len(inputs)

        for entity_relation in x['entity_relations']:
            entity_start = entity_relation['chunk_start']
            entity_end = entity_relation['chunk_end']

            if summary_token_ws[entity_end - 1][-1] == ' ':
                entity_mask = '<mask> '
            else:
                entity_mask = '<mask>'

            inp = []
            inp.extend(summary_token_ws[:entity_start])
            inp.append(entity_mask)
            inp.extend(summary_token_ws[entity_end:])
            inputs.append(''.join(inp))

            for head in entity_relation['heads']:
                head_start = head['chunk_start']
                head_end = head['chunk_end']

                if summary_token_ws[head_end - 1][-1] == ' ':
                    head_mask = '<mask> '
                else:
                    head_mask = '<mask>'

                inp = []
                inp.extend(summary_token_ws[:head_start])
                inp.append(head_mask)
                inp.extend(summary_token_ws[head_end:])
                inputs.append(''.join(inp))

                if head_start < entity_start:
                    if head_end == entity_start:
                        inp = []
                        inp.extend(summary_token_ws[:head_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:])
                        inputs.append(''.join(inp))
                    else:
                        inp = []
                        inp.extend(summary_token_ws[:head_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:entity_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:])
                        inputs.append(''.join(inp))
                else:
                    if entity_end == head_start:
                        inp = []
                        inp.extend(summary_token_ws[:entity_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:])
                        inputs.append(''.join(inp))
                    else:
                        inp = []
                        inp.extend(summary_token_ws[:entity_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:head_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:])
                        inputs.append(''.join(inp))

            for head in entity_relation['children']:
                head_start = head['chunk_start']
                head_end = head['chunk_end']

                if summary_token_ws[head_end - 1][-1] == ' ':
                    head_mask = '<mask> '
                else:
                    head_mask = '<mask>'

                inp = []
                inp.extend(summary_token_ws[:head_start])
                inp.append(head_mask)
                inp.extend(summary_token_ws[head_end:])
                inputs.append(''.join(inp))

                if head_start < entity_start:
                    if head_end == entity_start:
                        inp = []
                        inp.extend(summary_token_ws[:head_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:])
                        inputs.append(''.join(inp))
                    else:
                        inp = []
                        inp.extend(summary_token_ws[:head_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:entity_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:])
                        inputs.append(''.join(inp))
                else:
                    if entity_end == head_start:
                        inp = []
                        inp.extend(summary_token_ws[:entity_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:])
                        inputs.append(''.join(inp))
                    else:
                        inp = []
                        inp.extend(summary_token_ws[:entity_start])
                        inp.append(entity_mask)
                        inp.extend(summary_token_ws[entity_end:head_start])
                        inp.append(head_mask)
                        inp.extend(summary_token_ws[head_end:])
                        inputs.append(''.join(inp))

        with torch.no_grad():
            new_summary = bart.fill_mask(inputs, topk=args.topk, beam=5, match_source_len=False)
        new_entity_summary = [[xxx[0] for xxx in xx] for xx in new_summary[:num_entity]]  # MaskEnt
        new_rel_summary = [[xxx[0] for xxx in xx] for xx in new_summary[num_entity:]]  # MaskRel
        new_summaries.append({'summary': summary, 'new_ent_summary': new_entity_summary, 'new_rel_summary': new_rel_summary})

    with open(args.out, 'w') as f:
        for new_summary in new_summaries:
            f.write(json.dumps(new_summary) + '\n')


if __name__ == '__main__':
    main()
