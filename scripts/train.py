import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import StackDataset
from model.mlp import MLP
from utils.data_utils import make_path
from utils.train_utils import make_log_dir
from itertools import cycle
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(name: 'str',
          steps: int = 100000,
          batch_size: int = 32,
          lr: float = 3e-4,
          weight_decay: float = 1e-6,
          log_dir: str = 'logs',
          log_freq: int = 1000,
          seed: int = 42):

    torch.manual_seed(seed)
    make_log_dir(log_dir, name)

    tb_path = make_path(log_dir, name, 'tensorboard')
    writer = SummaryWriter(tb_path)
    ckpt_path = make_path(log_dir, name, 'checkpoints')

    train_loader = DataLoader(StackDataset(train=True), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(StackDataset(train=False),
                            batch_size=batch_size, shuffle=False, num_workers=0)
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = nn.MSELoss()

    iter_loader = iter(train_loader)
    for step in tqdm(range(steps)):
        try:
            ins, obs, actions = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            ins, obs, actions = next(iter_loader)

        ins = ins.to(device)
        obs = obs.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()
        mean, sigma = model(ins, obs)
        noise = torch.randn_like(sigma)
        actions_pred = mean + sigma * noise
        loss = criterion(actions_pred, actions)
        loss.backward()
        optimizer.step()

        if step % log_freq == 0:
            writer.add_scalar('train/loss', loss, step)
            model.eval()
            val_loss = validate(model, val_loader, criterion)
            writer.add_scalar('val/loss', val_loss, step)
            model.train()
            torch.save(model.state_dict(), 
                       os.path.join(ckpt_path, f'{step}.pt'))

    writer.flush()
    writer.close()


def validate(model, loader, criterion):
    with torch.no_grad():
        batch_num = 0
        total_loss = 0
        for ins, obs, actions in loader:
            ins = ins.to(device)
            obs = obs.to(device)
            actions = actions.to(device)

            mean, sigma = model(ins, obs)
            noise = torch.randn_like(sigma)
            actions_pred = mean + sigma * noise
            loss = criterion(actions_pred, actions)
            total_loss += loss.item()
            batch_num += 1

        return total_loss / batch_num


if __name__ == '__main__':
    train('mlp')
