python dump_dict.py
bash scripts/prepare-wmt14en2de-mbart.sh
python train.py -a transformer_wmt_en_de wmt14_en_de_toy_processed --max-sentences 1 --lr 0.00005 --max-epoch 50 --no-epoch-checkpoints
bash scripts/run2.sh
