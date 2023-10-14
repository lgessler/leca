import sys
from transformers import MBart50Tokenizer
from tqdm import tqdm


def main(infile, outfile):
    t = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
    with open(infile) as f:
        s = f.read()
    lines = s.strip().split("\n")

    with open(outfile, 'w') as f:
        for line in tqdm(lines):
            f.write(" ".join(t.tokenize(line)) + "\n")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python mbart_tokenize.py SRCFILE OUTFILE", file=sys.stderr)
        exit(1)
    main(sys.argv[1], sys.argv[2])
    

