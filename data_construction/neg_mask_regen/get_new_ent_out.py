import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filter_jsonl')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    others = []
    with open(args.filter_jsonl) as f, open(args.out_prefix + '.raw_target', 'w') as ftgt, \
            open(args.out_prefix + '.other', 'w') as fother:
        for i, line in enumerate(f):
            x = json.loads(line)
            if 'new_entity_summaries' not in x:
                continue
            ori_summary = x['ori_summary']
            new_entity_summaries = x['new_entity_summaries']
            for summary_and_new_entities in new_entity_summaries:
                if summary_and_new_entities['summary'].strip() == ori_summary.strip():
                    continue
                ftgt.write(summary_and_new_entities['summary'] + '\n')
                fother.write(str(i) + '\n')
                others.append(str(i))


if __name__ == '__main__':
    main()