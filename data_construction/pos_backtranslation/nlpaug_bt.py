import nlpaug.augmenter.word as naw
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    back_translation_aug = naw.BackTranslationAug(
        from_model_name='transformer.wmt19.en-de',
        to_model_name='transformer.wmt19.de-en',
        device='cuda'
    )

    with open(args.in_file) as f:
        lines = [line.strip() for line in f]

    error_line = []
    new_lines = []
    for i, line in enumerate(lines):
        if len(line.strip().split()) > 800:
            error_line.append(i)
            new_lines.append(line[:800])
        else:
            new_lines.append(line)

    print(error_line)
    error_line = set(error_line)

    with open(args.out_prefix + '.raw_target', 'w') as f, open(args.out_prefix + '.other', 'w') as fother:
        with torch.no_grad():
            for batch_start in range(0, len(new_lines), 64):
                for i, result in enumerate(back_translation_aug.augment(new_lines[batch_start:batch_start+64])):
                    if batch_start + i in error_line:
                        continue
                    f.write(result + '\n')
                    fother.write(str(batch_start + i) + '\n')


if __name__ == '__main__':
    main()

