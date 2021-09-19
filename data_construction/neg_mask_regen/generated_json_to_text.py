import argparse
import json
from tqdm import tqdm


def align_ws(old_token, new_token):
    # Align trailing whitespaces between tokens
    if old_token[-1] == new_token[-1] == " ":
        return new_token
    elif old_token[-1] == " ":
        return new_token + " "
    elif new_token[-1] == " ":
        return new_token[:-1]
    else:
        return new_token


def process_one(i, json_dict, generated_texts):
    ori_summ = json_dict['summary']

    text_outs = []

    for ent_i, (ori_entity, new_entity_summaries) in enumerate(zip(json_dict['entitys'], generated_texts['new_ent_summary'])):
        valid_entity_summaries = []
        for new_entity_summary in set(new_entity_summaries[:3]):
            if new_entity_summary == ori_summ:
                continue
            valid_entity_summaries.append(new_entity_summary)
            break

        text_outs.append((valid_entity_summaries, (i, 0, f'entity_{ent_i}', len(valid_entity_summaries))))

    if json_dict['entity_relations']:
        j = 0
        for rel_i, x in enumerate(json_dict['entity_relations']):
            num_heads = len(x['heads'])
            num_children = len(x['children'])

            j += 1  # entity chunk
            for head_i in range(num_heads):
                j += 1  # only head
                valid_summaries = []
                for new_rel_summary in set(generated_texts['new_rel_summary'][j][:3]):
                    if new_rel_summary == ori_summ:
                        continue
                    valid_summaries.append(new_rel_summary)
                    break
                j += 1  # both chunk and head
                text_outs.append((valid_summaries, (i, 0, f'relation_{rel_i}_head{head_i}', len(valid_summaries))))

            for head_i in range(num_children):
                j += 1  # only child
                valid_summaries = []
                for new_rel_summary in set(generated_texts['new_rel_summary'][j][:3]):
                    if new_rel_summary == ori_summ:
                        continue
                    valid_summaries.append(new_rel_summary)
                    break
                j += 1  # both chunk and child
                text_outs.append((valid_summaries, (i, 0, f'relation_{rel_i}_child{head_i}', len(valid_summaries))))

    return text_outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    parser.add_argument('generated_file')
    parser.add_argument('out_file')
    parser.add_argument('new_other_file')
    args = parser.parse_args()

    generated_texts = []
    with open(args.generated_file) as f:
        for line in f:
            generated_texts.append(json.loads(line))

    with open(args.jsonl_file) as f:
        lines = f.readlines()

    results = []
    for i, line in tqdm(enumerate(lines)):
    # for i, line in enumerate(f):
        x = json.loads(line)
        if len(x['entitys']) + len(x['entity_relations']) > 0:
            results.append(process_one(i, x, generated_texts[i]))

    with open(args.out_file, 'w') as f, open(args.new_other_file, 'w') as fother:
        for result in results:
            for beams, other in result:
                for beam in beams:
                    f.write(beam + '\n')
                fother.write(' '.join([str(xx) for xx in other]) + '\n')


if __name__ == '__main__':
    main()
