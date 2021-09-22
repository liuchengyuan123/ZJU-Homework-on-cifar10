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


def build_dataset():
    data_path = './data/'
    d, l = [], []
    data_name = data_path + 'test_batch'
    data_d, data_l = read_data(data_name)
    d.append(data_d)
    l.append(data_l)

    d = np.concatenate(d, axis=0)
    l = np.concatenate(l, axis=0)

    test_dataset = CifarDataset(d, l)

    return test_dataset


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet50 = ResNet50WithDropout()
        self.softmax = nn.Softmax()

    def forward(self, x):
        return self.softmax(self.resnet50(x))


def build_model(device, args):
    model = Model()

    if device:
        model = model.to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoint/Wed-Sep-22-07:57:06-2021/best.pt', help='checkpoint path')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('using', device)
    
    test_dataset = build_dataset()
    
    model = build_model(device, args)
    tot = 0
    correct = 0
    loss = 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    t = tqdm(test_dataloader, total=len(test_dataset) // args.batch_size)

    criterion = nn.CrossEntropyLoss()
    if device:
        criterion = criterion.to(device)
    
    model.eval()
    for batch_x, batch_y in t:
        if device:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        prediction = model(batch_x)
        cur_loss = criterion(prediction, batch_y)

        pred = prediction.argmax(dim=1)
        correct += torch.eq(pred, batch_y).float().sum().item()

        tot += batch_x.size()[0]
        
        loss += cur_loss.item() * batch_x.size()[0]
    
    print(f'total: {tot}, loss: {loss / tot}, accuracy: {correct / tot}')
