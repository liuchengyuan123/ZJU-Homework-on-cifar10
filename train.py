from os import read
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import argparse
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# from Model.Resnet import ResNet50
from Model.ResNetWithDropOut import ResNet50WithDropout


def read_data(path):
    f = open(path, 'rb')
    data = pickle.load(f, encoding='bytes')
    return data[b'data'].reshape(-1, 3, 32, 32).astype('float'),\
        np.array(data[b'labels']).reshape(-1).astype('float')


class CifarDataset(Dataset):
    def __init__(self, d, l) -> None:
        self.x_data = torch.FloatTensor(d)
        self.y_data = torch.LongTensor(l)
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def build_datasets(train_rate=0.8):
    train_size = int(5 * train_rate)
    assert train_size < 5
    data_path = './data/'
    train_d, train_l = [], []
    for i in range(train_size):
        data_name = data_path + f'data_batch_{i + 1}'
        d, l = read_data(data_name)
        train_d.append(d)
        train_l.append(l)

    val_d, val_l = [], []
    for i in range(train_size, 5):
        data_name = data_path + f'data_batch_{i + 1}'
        d, l = read_data(data_name)
        val_d.append(d)
        val_l.append(l)

    train_d = np.concatenate(train_d, axis=0)
    train_l = np.concatenate(train_l, axis=0)
    val_d = np.concatenate(val_d, axis=0)
    val_l = np.concatenate(val_l, axis=0)

    train_dataset = CifarDataset(train_d, train_l)
    val_dataset = CifarDataset(val_d, val_l)

    return train_dataset, val_dataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet50 = ResNet50WithDropout()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.resnet50(x))


def build_model(device, args):
    model = Model()
    criterion = nn.CrossEntropyLoss()

    if device:
        model = model.to(device)
        criterion = criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, criterion, optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int,
                        default=200, help='number epochs')
    parser.add_argument('--display', type=int, default=1,
                        help='display batch number')

    args = parser.parse_args()

    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('using', device)

    model, criterion, optimizer = build_model(device, args)

    # NOTE base line
    saved_model_path = 'checkpoint/Tue-Sep-21-22:52:22-2021/best.pt'
    # saved_model_path = ''
    if saved_model_path:
        model.load_state_dict(torch.load(saved_model_path))

    checkpoint_path = args.checkpoint
    import os

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    import time
    time_stamp = time.ctime(time.time()).replace(' ', '-')
    checkpoint_path = os.path.join(checkpoint_path, time_stamp)
    os.mkdir(checkpoint_path)

    loss_history = []
    acc_history = []
    train_history = []
    train_acc_history = []

    batch_size = args.batch_size

    train_dataset, val_dataset = build_datasets()
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)

    bst_loss = 1e15
    bst_acc = 0
    print('start training')
    for epoch in range(args.epochs):
        loss_sum = []
        model.train()
        tqdm_train = tqdm(enumerate(train_dataloader), total=len(train_dataset) // batch_size)
        tqdm_train.set_description(f'epoch {epoch + 1}')
        for batch_idx, (x, label) in tqdm_train:
            optimizer.zero_grad()
            if device:
                x, label = x.to(device), label.to(device)
            prediction = model(x)
            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()

            pred = prediction.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            acc = correct / pred.size()[0]
            train_history.append(loss.item())
            train_acc_history.append(acc)

            if (batch_idx + 1) % args.display == 0:
                print(batch_idx, 'loss: ', loss.item(), 'acc: ', acc)
                loss_sum.append(loss.item())
        train_loss = sum(loss_sum) / len(loss_sum)

        model.eval()
        loss_sum = []
        with torch.no_grad():
            # test
            total_correct = 0
            total_num = 0
            tqdm_val = tqdm(val_dataloader, total=len(val_dataset) // batch_size)
            tqdm_val.set_description(f'running validation')
            for x, label in tqdm_val:
                if device:
                    x, label = x.to(device), label.to(device)
                prediction = model(x)
                loss = criterion(prediction, label)
                loss_sum.append(loss.item())

                pred = prediction.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += x.size(0)

            acc = total_correct / total_num
            cur_loss = sum(loss_sum) / len(loss_sum)
        
        print(f'epoch {epoch + 1}, train loss {train_loss}')

        if cur_loss < bst_loss:
            save_path = os.path.join(checkpoint_path, 'best.pt')
            print(f'epoch {epoch + 1}: loss from {bst_loss} to {cur_loss}, '
                  f'accuracy from {bst_acc} to {acc}')

            print(f'saving to {save_path}')
            torch.save(model.state_dict(), open(save_path, 'wb'))
            bst_loss = cur_loss
            bst_acc = acc
        loss_history.append(cur_loss)
        acc_history.append(acc)

    # save final model
    torch.save(model.state_dict(), open(os.path.join(checkpoint_path, 'final.pt'), 'wb'))
    
    plt.clf()
    plt.plot(train_history)
    plt.savefig(os.path.join(checkpoint_path, 'train_loss.png'))

    plt.clf()
    plt.plot(train_acc_history)
    plt.savefig(os.path.join(checkpoint_path, 'train_accuracy.png'))

    plt.clf()
    plt.plot(loss_history)
    plt.savefig(os.path.join(checkpoint_path, 'val_loss.png'))

    plt.clf()
    plt.plot(acc_history)
    plt.savefig(os.path.join(checkpoint_path, 'val_acc.png'))

    print(f'achieved best accuarcy {bst_acc}, best loss {bst_loss}')

