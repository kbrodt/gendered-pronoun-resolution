import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def insert_tag(row):
    to_be_inserted = sorted([
        (row['A-offset'], ' [A] '),
        (row['B-offset'], ' [B] '),
        (row['Pronoun-offset'], ' [P] ')
    ], key=lambda x: x[0], reverse=True)
    text = row['Text']
    for offset, tag in to_be_inserted:
        text = text[:offset] + tag + text[offset:]
        
    return text


def tokenize(text, tokenizer):
    entries = {}
    final_tokens = []
    for token in tokenizer.tokenize(text):
        if token in ('[A]', '[B]', '[P]'):
            entries[token] = len(final_tokens)
            continue
        final_tokens.append(token)

    return final_tokens, (entries['[A]'], entries['[B]'], entries['[P]'])


class GAPDataset(Dataset):
    def __init__(self, df, tokenizer, labeled=True):
        self.labeled = labeled
        
        if labeled:
            tmp = df[['A-coref', 'B-coref']].copy()
            tmp['Neither'] = ~(df['A-coref'] | df['B-coref'])
            self.y = tmp.values.astype('bool')

        self.offsets, self.tokens = [], []
        for _, row in df.iterrows():
            text = insert_tag(row)
            tokens, offsets = tokenize(text, tokenizer)
            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + tokens + ['[SEP]']))
        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        if self.labeled:
            return self.tokens[idx], self.offsets[idx], self.y[idx]
        
        return self.tokens[idx], self.offsets[idx]
    
    
def collate_examples(batch, truncate_len=512):
    transposed = list(zip(*batch))
    
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = row[:truncate_len]
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    
    offsets = torch.stack([
        torch.tensor(x, dtype=torch.int64) for x in transposed[1]
    ], dim=0) + 1
    
    if len(transposed) == 2:
        return token_tensor, offsets
    
    one_hot_labels = torch.stack([
        torch.from_numpy(x.astype('uint8')) for x in transposed[2]
    ], dim=0)
    _, labels = one_hot_labels.max(dim=1)
    
    return token_tensor, offsets, labels


def create_data_loader(df, tokenizer, labeled=True, batch_size=16, num_workers=8, shuffle=False):
    ds = GAPDataset(df, tokenizer, labeled=labeled)
    return DataLoader(
        ds,
        collate_fn=collate_examples,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
    )
