import pandas as pd
import torch

# load CSV file into pandas DataFrame
df = pd.read_csv('musiccaps-public.csv')

# select a particular column to tokenize
column_name = 'caption'
texts = df[column_name].tolist()

# define tokenizer function
tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')


def tokenize(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return tokens


# tokenize texts
tokenized_texts = [tokenize(text) for text in texts]

# save tokenized texts in .pt file
torch.save(tokenized_texts, 'tokenized_texts.pt')
