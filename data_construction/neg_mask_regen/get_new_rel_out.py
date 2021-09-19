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
            if 'new_relation_summarys' not in x:
                continue
            ori_summary = x['ori_summary']
            new_relation_summaries = x['new_relation_summarys']
            for summary_and_new_relations in new_relation_summaries:
                if summary_and_new_relations['summary'] != ori_summary:
                    ftgt.write(summary_and_new_relations['summary'] + '\n')
                    fother.write(str(i) + '\n')
                    others.append(str(i))


if __name__ == '__main__':
    main()