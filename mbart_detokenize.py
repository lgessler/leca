import sys
from transformers import MBart50Tokenizer
from tqdm import tqdm


def main(infile, outfile):
    t = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
    with open(infile) as f:
        s = f.read()
    lines = s.strip().split("\n")
    v = t.get_vocab()

    with open(outfile, 'w') as f:
        for line in tqdm(lines):
            s = t.decode([v[wp] for wp in line.split(" ")])
            f.write(s + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python mbart_detokenize.py SRCFILE OUTFILE", file=sys.stderr)
        exit(1)
    main(sys.argv[1], sys.argv[2])
    

