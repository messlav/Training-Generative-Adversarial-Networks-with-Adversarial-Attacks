# Training SNGAN with FGSM

# References of SNGAN https://github.com/w86763777/pytorch-gan-collections

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
# model
class Generator(nn.Module):
    def __init__(self, z_dim, M=4):
        super().__init__()
        self.M = M
        self.linear = nn.Linear(z_dim, M * M * 512)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z)
        x = x.view(x.size(0), -1, self.M, self.M)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, M=32):
        super().__init__()
        self.M = M

        self.main = nn.Sequential(
            # M
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 2
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            # M / 8
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

        self.linear = nn.Linear(M // 8 * M // 8 * 512, 1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.normal_(m.weight, std=0.02)
                torch.nn.init.zeros_(m.bias)
                torch.nn.utils.spectral_norm(m)

    def forward(self, x, *args, **kwargs):
        x = self.main(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


class Hinge(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = F.relu(1 - pred_real).mean()
            loss_fake = F.relu(1 + pred_fake).mean()
            wandb.log({"train/loss_real": loss_real,
                      "train/loss_fake": loss_fake})
            return loss_real + loss_fake
        else:
            loss = -pred_real.mean()
            return loss

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

# Training
def train_gan_w_fgsm(dataloader, net_D, net_G, num_epochs, loss_fn, optim_G, optim_D, sched_G, sched_D, record, run_name=None):        
    if record:
        wandb.init(
        project="SNGAN_w_FGSM",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "betas": betas,
            "sheduler": "LambdaLR",
            "sheduler step": "1 - step / 100000",
            "sample step": 500,
            "FGSM epsilon": epsilon,
            "FGSM chance": fgsm_chance,
            "start FGSM epoch": start_fgsm_epoch
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
        
    # Training Loop    
    print("Starting Training Loop...")
    start = time.time()
    for epoch in range(1, num_epochs + 1):
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader)):
            # Discriminator
            with torch.no_grad():
                z = torch.randn(128, 100).to(device)
                fake = net_G(z).detach()
            real = data[0].to(device)
            real.requires_grad = True
            fake.requires_grad = True
            net_D_real = net_D(real)
            net_D_fake = net_D(fake)
            loss_D = loss_fn(net_D_real, net_D_fake)
            if epoch >= start_fgsm_epoch and np.random.random() < fgsm_chance:
                # FGSM
                real_grad = torch.autograd.grad(loss_D, [real])[0]
                fake_grad = torch.autograd.grad(loss_D, [fake])[0]
                perturbed_real = fgsm_attack(real, epsilon, real_grad)
                perturbed_fake = fgsm_attack(fake, epsilon, fake_grad)
                net_D_real = net_D(perturbed_real)
                net_D_fake = net_D(perturbed_fake)
                loss_D = loss_fn(net_D_real, net_D_fake)
            if record:
                wandb.log({"train/net_D_real": net_D_real,
                            "train/net_D_fake": net_D_fake})
            
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            
            # Generator
            z = torch.randn(128 * 2, 100).to(device)
            if epoch >= start_fgsm_epoch and np.random.random() < fgsm_chance:
                # FGSM
                fake = net_G(z)
                net_D_fake = net_D(fake)
                loss_G = loss_fn(net_D_fake)
                fake_grad = torch.autograd.grad(loss_G, [fake])[0]
                perturbed_fake_G = fgsm_attack(fake, epsilon, fake_grad)
                loss_G = loss_fn(net_D(perturbed_fake_G))
            else:
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
            fake = net_G(sample_z).detach()
            grid = (make_grid(fake.cpu()) + 1) / 2
            images = wandb.Image(grid, caption="after epoch {}".format(epoch))
            wandb.log({"samples": images})
            
            if epoch >= start_fgsm_epoch:
                # FGSM log
                grid = (make_grid(perturbed_real.cpu()) + 1) / 2
                images_real = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                grid = (make_grid(perturbed_fake.cpu()) + 1) / 2
                images_fake = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                grid = (make_grid(perturbed_fake_G.cpu()) + 1) / 2
                images_G = wandb.Image(grid, caption="last in epoch {}".format(epoch))
                
                wandb.log({
                    "FGSM on real in D": images_real,
                    "FGSM on fake in D": images_fake,
                    "FGSM in G": images_G
                })
            
            imgs = generate_imgs(
                    net_G, device, 100,
                    50000, batch_size)
            IS, FID = get_inception_score_and_fid(
                imgs, 'cifar10.train.npz', verbose=True)
            is_fid_imgs = {"train/IS": IS[0],
                          "train/IS_std": IS[1],
                          "train/FID:": FID}
            wandb.log(is_fid_imgs)

        if epoch % 5 == 0:
            clear_output()
        
    print('Training for batch size = {}, epochs = {} done for {:.1f} hours'.format(batch_size, num_epochs, (time.time() - start) / 60 / 60))
    wandb.finish()


set_random_seed(3407)

batch_size = 128
dataset = torchvision.datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Lambda(lambda x: x + torch.rand_like(x) / 128)
            ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=8, drop_last=True)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

net_G = Generator(100).to(device)
net_D = Discriminator().to(device)

loss_fn = Hinge()

lr = 0.0002
betas = [0.0, 0.9]
optim_G = torch.optim.Adam(net_G.parameters(), lr=lr, betas=betas)
optim_D = torch.optim.Adam(net_D.parameters(), lr=lr, betas=betas)
sched_G = torch.optim.lr_scheduler.LambdaLR(
    optim_G, lambda step: 1 - step / 100000)
sched_D = torch.optim.lr_scheduler.LambdaLR(
    optim_D, lambda step: 1 - step / 100000)

num_epochs = 250
fgsm_chance = 0.8
epsilon = 0.02
start_fgsm_epoch = int(num_epochs * 0.05)

sample_z = torch.randn(64, 100).to(device)


# run code
train_gan_w_fgsm(dataloader, net_D, net_G, num_epochs, loss_fn, optim_G, optim_D, sched_G, sched_D, record=True,
                 run_name="try3-fix-FGSM-chance-0.8")

