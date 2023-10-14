1. Create a python environment:

```
conda create --name leca python=3.8
conda activate leca
```

2. Install dependencies:

```
pip install -e .
pip install transformers sentencepiece "numpy<20" "torch<2"
```

3. Clone moses: `git clone https://github.com/moses-smt/mosesdecoder.git`

4. Prepare the mbart dict: `python dump_dict.py`

5. Acquire data and tokenize it with mbart: `bash scripts/prepare-wmt14en2de-mbart.sh`

6. (Optional): make a toy dataset for debugging

```
head -n 100 wmt14_en_de/train.en > wmt14_en_de/train_toy.en
head -n 100 wmt14_en_de/train.de > wmt14_en_de/train_toy.de
head -n 10 wmt14_en_de/train.en > wmt14_en_de/valid_toy.en
head -n 10 wmt14_en_de/train.en > wmt14_en_de/test_toy.en
head -n 10 wmt14_en_de/train.de > wmt14_en_de/valid_toy.de
head -n 10 wmt14_en_de/train.de > wmt14_en_de/test_toy.de
```

7. Preprocess data into fairseq's binary format:

```
python preprocess.py --workers 8 -s en -t de \
    --destdir wmt14_en_de/processed_data \
    --trainpref wmt14_en_de/train \
    --validpref wmt14_en_de/valid \
    --testpref wmt14_en_de/test   \
    --srcdict facebook__mbart-large-50 \
    --tgtdict facebook__mbart-large-50
# Or if you're using the toy dataset:
python preprocess.py --workers 8 -s en -t de \
    --destdir wmt14_en_de/processed_data \
    --trainpref wmt14_en_de/train_toy \
    --validpref wmt14_en_de/valid_toy \
    --testpref wmt14_en_de/test_toy   \
    --srcdict facebook__mbart-large-50 \
    --tgtdict facebook__mbart-large-50
```

8. Run the train/test pipeline:

```
bash scripts/run_modified.sh
# or 
bash scripts/run_toy.sh
```
