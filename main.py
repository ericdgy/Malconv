import torch
import pandas as pd
from dataset import ExeDataset, init_loader
from model import MalConv
from train import train_model

train_data_path = 'data/train_sections/'
train_label_path = 'data/train-label-headers.csv'
valid_data_path = 'data/valid_sections/'
valid_label_path = 'data/valid-label-headers.csv'

tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
tr_label_table.index = tr_label_table.index.str.upper()
tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
val_label_table.index = val_label_table.index.str.upper()
val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
val_table = val_label_table.groupby(level=0).last()
del val_label_table
tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

print('Training Set:')
print('\tTotal', len(tr_table), 'files')
print('\tMalware Count :', tr_table['ground_truth'].value_counts().iloc[1])
print('\tGoodware Count:', tr_table['ground_truth'].value_counts().iloc[0])

print('Validation Set:')
print('\tTotal', len(val_table), 'files')
print('\tMalware Count :', val_table['ground_truth'].value_counts().iloc[1])
print('\tGoodware Count:', val_table['ground_truth'].value_counts().iloc[0])

first_n_byte = 2000000
window_size = 500
batch_size = 16

train_dataset = ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth), first_n_byte)
valid_dataset = ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), first_n_byte)

train_loader, valid_loader = init_loader(train_dataset, batch_size)
valid_loader = init_loader(valid_dataset, batch_size)[1]

model = MalConv(input_length=first_n_byte, window_size=window_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

if torch.cuda.is_available():
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")


criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)


epochs = 10

best_model = train_model(model, criterion, optimizer, device, epochs, train_loader, valid_loader)
