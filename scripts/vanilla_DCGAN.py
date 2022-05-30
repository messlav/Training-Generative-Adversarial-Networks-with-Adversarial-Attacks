# My realization of DCGAN with logging 

# References https://github.com/w86763777/pytorch-gan-collections

# PREWORK

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import trange
from torch import nn
import random
import time
from IPython.display import clear_output
import torchvision.transforms as T
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.utils import make_grid, save_image
import os
from pytorch_gan_metrics import get_inception_score_and_fid
import wandb

wandb.login(key='ecbb7f54f997fbf05d9983bfc9847256eea55632') # you need to have wandb.ai free account

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def generate_imgs(net_G, device, z_dim=128, size=5000, batch_size=128):
    net_G.eval()
    imgs = []
    with torch.no_grad():
        for start in trange(0, size, batch_size,
                            desc='Evaluating', ncols=0, leave=False):
            end = min(start + batch_size, size)
            z = torch.randn(end - start, z_dim).to(device)
            imgs.append(net_G(z).cpu())
    net_G.train()
    imgs = torch.cat(imgs, dim=0)
    imgs = (imgs + 1) / 2
    return imgs

# REALIZATION

image_size = 64
batch_size = 128
dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=8, drop_last=True)

set_random_seed(3407)
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

def train_gan(dataloader, net_D, net_G, num_epochs, loss_fn, optim_G, optim_D, sched_G, sched_D, record, run_name=None):
    # Training Loop    
    print("Starting Training Loop...")
    start = time.time()
        
    if record:
        wandb.init(
        project="DCGAN",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "betas": betas,
            "sheduler": "LambdaLR",
            "sheduler step": "1 - step / 100000",
            "sample step": 500
            })
        if run_name is not None:
            wandb.run.name = run_name
        
        fake = net_G(sample_z).cpu()
        grid = (make_grid(fake) + 1) / 2
        real, _ = next(iter(dataloader))
        grid2 = (make_grid(real[:64]) + 1) / 2
        images = wandb.Image(grid2, caption="real")
        wandb.log({"samples": images})
        images = wandb.Image(grid, caption="before training")
        wandb.log({"samples": images})
    
    for epoch in range(1, num_epochs + 1):
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc=f'Training {epoch}/{num_epochs}')):
            # Discriminator
            with torch.no_grad():
                z = torch.randn(128, 100).to(device)
                fake = net_G(z).detach()
            
            real = data[0].to(device)
            net_D_real = net_D(real)
            net_D_fake = net_D(fake)
            loss_D = loss_fn(net_D_real, net_D_fake)

            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            
            # Generator
            z = torch.randn(128 * 2, 100).to(device)
            loss_G = loss_fn(net_D(net_G(z)))

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            sched_G.step()
            sched_D.step()
            
            if record:
                metrics = {"train/generator_loss": loss_G, 
                           "train/discriminator_loss": loss_D}
                wandb.log(metrics)
            
        print('loss_g = {:.3f}, loss_d = {:.3f}'.format(loss_G, loss_D))
        if record:
            fake = net_G(sample_z).cpu()
            grid = (make_grid(fake) + 1) / 2
            images = wandb.Image(grid, caption="after epoch {}".format(epoch))
            imgs = generate_imgs(
                    net_G, device, 100,
                    50000, batch_size)
            IS, FID = get_inception_score_and_fid(
                imgs, 'cifar10.train.npz', verbose=True)
            is_fid_imgs = {"train/IS": IS[0],
                          "train/IS_std": IS[1],
                          "train/FID:": FID,
                          "samples": images}
            wandb.log(is_fid_imgs)

        
        if epoch % 5 == 0:
            clear_output()
        
    print('Training for batch size = {}, epochs = {} done for {:.1f} minutes'.format(batch_size, num_epochs, (time.time() - start) / 60))
    wandb.finish()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 2, 1, 0, bias=False),  # 4, 4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z.view(-1, 100, 1, 1))
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 64
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            # 16
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            # 8
            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(512)
            # 4
        )

        self.linear = nn.Linear(32 // 16 * 32 // 16 * 512, 1)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
    
class BCEWithLogits(nn.BCEWithLogitsLoss):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = super().forward(pred_real, torch.ones_like(pred_real))
            loss_fake = super().forward(pred_fake, torch.zeros_like(pred_fake))
            return loss_real + loss_fake
        else:
            loss = super().forward(pred_real, torch.ones_like(pred_real))
            return loss


net_G = Generator().to(device)
net_D = Discriminator().to(device)

loss_fn = BCEWithLogits()

lr = 0.0002
betas = [0.5, 0.9]
optim_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=betas)
optim_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=betas)
sched_G = torch.optim.lr_scheduler.LambdaLR(
    optim_G, lambda step: 1 - step / 100000)
sched_D = torch.optim.lr_scheduler.LambdaLR(
    optim_D, lambda step: 1 - step / 100000)

sample_z = torch.randn(64, 100).to(device)

num_epochs = 250

train_gan(dataloader, net_D, net_G, num_epochs, loss_fn, optim_G, optim_D, sched_G, sched_D, record=True,
          run_name="Vanilla-DCGAN")

