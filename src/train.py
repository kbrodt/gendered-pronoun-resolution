from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import StratifiedKFold
import torch
import tqdm

from data_loader import create_data_loader
from model import GAP
from drop import TO_DROP


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def get_label(row):
    if row['A-coref']:
        return 0
    if row['B-coref']:
        return 1
    
    return 2


def correct(df, to_correct_df):
    df['label'] = pd.merge(df, to_correct_df,
                           how='left',
                           left_on='ID',
                           right_on='ID')['label']
    df.loc[df.label == 'A', 'A-coref'] = True
    df.loc[df.label == 'A', 'B-coref'] = False
    
    df.loc[df.label == 'B', 'B-coref'] = True
    df.loc[df.label == 'B', 'A-coref'] = False
    
    df.loc[df.label == 'N', 'A-coref'] = False
    df.loc[df.label == 'N', 'B-coref'] = False


def epoch_step(loader, desc, model, criterion, device, opt=None, n_gpu=1):
    loc_loss = n = 0
    with tqdm.tqdm(loader,
                     mininterval=2,
                     desc=desc,
                     leave=False) as pbar:
        for batch in pbar:
            logits = model(*[b.to(device) for b in batch[:2]])
            loss = criterion(logits, batch[-1].to(device))
            if n_gpu > 1:
                loss = loss.sum()

            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loc_loss += loss.item()
            n += batch[-1].size(0)

            pbar.set_postfix(**{'loss': f'{loc_loss/n:.3}'})
    
    return loc_loss/n


def predict(loader, model, device, criterion=None):
    is_train = criterion is not None
    if is_train:
        loss = n = 0
    preds, true_labels = [], []
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(loader, desc='[ Testing.. ]', mininterval=2, leave=False) as pbar:
            for batch in pbar:
                logits = model(*[b.to(device) for b in batch[:2]])
                if is_train:
                    loss += criterion(logits, batch[-1].to(device)).item()
                    n += batch[-1].size(0)
                    true_labels.extend(batch[-1].cpu().numpy())

                pred = torch.softmax(logits, dim=1).cpu().numpy()
                preds.extend(pred)
    preds = np.array(preds)
    
    if is_train:
        return preds, np.array(true_labels), loss/n

    return preds


def main():
    RANDOM_SEED = 314159
    set_random_seed(RANDOM_SEED)
    PATH_TO_DATA = Path('./data')
    
    train = pd.read_csv(PATH_TO_DATA / 'gap-test.tsv', sep='\t')
    dev = pd.read_csv(PATH_TO_DATA / 'gap-validation.tsv', sep='\t')
    test = pd.read_csv(PATH_TO_DATA / 'gap-development.tsv', sep='\t')
    to_test = pd.read_csv(PATH_TO_DATA / 'test_stage_2.tsv.zip', sep='\t')
    
    print(f'\n# Train: {len(train)}'
          f'\n# Dev: {len(dev)}'
          f'\n# Test: {len(test)}'
          f'\n# To Test: {len(to_test)}')
    
    # drop
    train = train[~train.ID.isin(TO_DROP.ID.values)].copy()
    dev = dev[~dev.ID.isin(TO_DROP.ID.values)].copy()
    test = test[~test.ID.isin(TO_DROP.ID.values)].copy()
    
    train['label'] = train.apply(get_label, axis=1)
    dev['label'] = dev.apply(get_label, axis=1)
    test['label'] = test.apply(get_label, axis=1)
    
    train = pd.concat([train, dev, test])
    
    print(f'\nAfter drop/correction # train+dev: {len(train)}'
          f'\nAfter drop/correction # test: {len(test)}')
    
    BERT_MODEL = 'bert-large-cased'
    do_lower_case = False
    print(f'\nUsing {BERT_MODEL} with lower_case={do_lower_case}')
    
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=do_lower_case,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")
    )

    # These tokens are not actually used, so we can assign arbitrary values.
    tokenizer.vocab["[A]"] = -1
    tokenizer.vocab["[B]"] = -1
    tokenizer.vocab["[P]"] = -1
    
    device = torch.device('cuda')
    n_gpu = torch.cuda.device_count()
    print(f'\n# GPU used: {n_gpu}')
    
    batch_size = 16*n_gpu
    print(f'\nBatch size per 1 GPU: {batch_size//n_gpu}')
    
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_SEED)

    to_test_preds = []
    path_to_models = Path('models_drop_all')
    for exp_num, (train_index, valid_index) in enumerate(skf.split(train, train['label'])):
        path_to_exp = path_to_models / f'exp_{exp_num + 1:0>2}'
        if not path_to_exp.exists():
            path_to_exp.mkdir(parents=True)
        print(f'\nFold {exp_num + 1} in {path_to_exp}')

        bert_out = 1024  # 768
        model = GAP(bert_model=BERT_MODEL, input_dim=bert_out, hidden_dim=2*1024, p=0.8, last_layers=8, fine_tune=False).to(device)

        train_loader = create_data_loader(train.iloc[train_index], tokenizer, labeled=True, batch_size=batch_size, shuffle=True)
        dev_loader = create_data_loader(train.iloc[valid_index], tokenizer, labeled=True, batch_size=batch_size, shuffle=False)
        to_test_loader = create_data_loader(to_test, tokenizer, labeled=False, batch_size=batch_size, shuffle=False)

        criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

        n_epochs = 20
        train_losses, dev_losses, test_losses = [], [], []
        best_dev_loss = float('+inf')
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_losses.append(epoch_step(train_loader,
                                           desc=f'[ Training.. {epoch}/{n_epochs} ]',
                                           model=model, criterion=criterion, device=device, opt=opt, n_gpu=n_gpu))

            with torch.no_grad():
                model.eval()
                dev_losses.append(epoch_step(dev_loader,
                                             desc=f'[ Validating.. {epoch}/{n_epochs} ]',
                                             model=model, criterion=criterion, device=device, n_gpu=n_gpu))
                if dev_losses[-1] < best_dev_loss:
                    best_dev_loss = dev_losses[-1]
                    torch.save({
                        'model': model.state_dict(),
                        'opt': opt.state_dict(),
                        'epoch': epoch,
                        'dev_loss': best_dev_loss,
                    }, path_to_exp / 'model.pt')

            plt.plot([np.nan] + train_losses[1:], label=f'train: {train_losses[-1]:.3}')
            plt.plot(dev_losses, label=f'dev:   {dev_losses[-1]:.3}')
            plt.legend()
            plt.xlabel('# epoch')
            plt.ylabel('xEntropy loss')
            plt.grid(ls='--')
            plt.savefig(path_to_exp / 'train_evolution.png')
            plt.close()

        print(f'\nDev : #epoch={np.argmin(dev_losses) + 1:>2},  loss={min(dev_losses)}')

        model.load_state_dict(torch.load(path_to_exp / 'model.pt')['model'])
        
        to_test_preds.append(predict(to_test_loader, model, device=device))
        print('-'*60)

    to_test_preds = np.mean(to_test_preds, axis=0)
    
    df_sub = pd.read_csv('sample_submission_stage_2.csv')
    df_sub[['A', 'B', 'NEITHER']] = pd.DataFrame(to_test_preds.clip(1e-4, 1 - 1e-4),
                                             columns=['A', 'B', 'NEITHER'],
                                             index=df_sub.index)
    df_sub.to_csv(path_to_models / 'submission.csv', index=False)


if __name__ == '__main__':
    main()
