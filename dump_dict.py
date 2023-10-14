from transformers import MBart50Tokenizer 
t = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
with open('facebook__mbart-large-50', 'w') as f:
    f.write("\n".join(f"{k} {v}" for k, v in t.get_vocab().items())+"\n")
