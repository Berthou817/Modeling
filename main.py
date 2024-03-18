import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from torch.utils.data.distributed import DistributedSampler
from bicyclegan import *
import util
import os
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from dataloader  import *
import argparse
import torch.distributed as dist
import os
if not os.path.exists("loss"):
    os.mkdir("loss")
else:
    os.rmdir("loss")
    os.mkdir("loss")
if not os.path.exists("model"):
    os.mkdir("model")
else:
    os.rmdir("model")
    os.mkdir("model")
if not os.path.exists("photo"):
    os.mkdir("photo")
else:
    os.rmdir("photo")
    os.mkdir("photo")

lr=0.0002
beta_1=0.5
beta_2=0.999
lambda_kl = 0.01
lambda_img = 10
lambda_z = 0.5
mse = nn.MSELoss().cuda()

def mse_loss(score, target=1):

    if target == 1:
        label = util.var(torch.ones(score.size()), requires_grad=False)
    elif target == 0:
        label = util.var(torch.zeros(score.size()), requires_grad=False)

    criterion = nn.MSELoss().cuda()
    loss = criterion(score, label.cuda())

    return loss

def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def lr_decay_rule(epoch, start_decay=100, lr_decay=100):
    decay_rate = 1.0 - (max(0, epoch - start_decay) / float(lr_decay))
    return decay_rate

D_cVAE = Discriminator().cuda()
D_cLR = Discriminator().cuda()
G = StyleGenerator().cuda()
E = Encoder().cuda()

optim_D_cVAE = optim.Adam(D_cVAE.parameters(), lr=lr, betas=(beta_1, beta_2))
optim_D_cLR = optim.Adam(D_cLR.parameters(), lr=lr, betas=(beta_1, beta_2))
optim_G = optim.Adam(G.parameters(), lr=lr, betas=(beta_1, beta_2))
optim_E = optim.Adam(E.parameters(), lr=lr, betas=(beta_1, beta_2))


def all_zero_grad():
    optim_D_cVAE.zero_grad()
    optim_D_cLR.zero_grad()
    optim_G.zero_grad()
    optim_E.zero_grad()
train_data  = trainset()
dataloader = DataLoader(train_data, batch_size=1)
epochs = 1000

GLoss_list = []
ELoss_list = []
DLoss_list = []
imgLoss_list = []
GALoss_list  = []
for ep in range(epochs):
    i = 0
    Gloss_list = []
    Eloss_list = []
    Dloss_list = []
    imgloss_list = []
    GAloss = []

    G.train()
    E.train()
    D_cLR.train()
    D_cVAE.train()
    for img,porosity in dataloader:
        i = i+1
        img = img.cuda()
        porosity = porosity.view(-1,256).cuda()
        ''' ----------------------------- 1. Train D ----------------------------- '''
        # Encoded latent vector
        mu, log_variance = E(img)
        std = torch.exp(log_variance / 2)
        encoded_z = (porosity * std) + mu
        # Generate fake image
        # encoded_z = torch.cat([encoded_z[1:15].view(1,15),flag])
        fake_img_cVAE = G(encoded_z)

        # Get scores and loss
        real_d_cVAE_1, real_d_cVAE_2 = D_cVAE(img)
        fake_d_cVAE_1, fake_d_cVAE_2 = D_cVAE(fake_img_cVAE)
        # mse_loss for LSGAN
        D_loss_cVAE_1 = mse_loss(real_d_cVAE_1, 1) + mse_loss(fake_d_cVAE_1, 0)
        D_loss_cVAE_2 = mse_loss(real_d_cVAE_2, 1) + mse_loss(fake_d_cVAE_2, 0)
        # Generate fake image
        # porosity = torch.cat([porosity[1:15].view(1,15),flag])
        fake_img_cLR = G(porosity)
        # Get scores and loss
        real_d_cLR_1, real_d_cLR_2 = D_cLR(img)
        fake_d_cLR_1, fake_d_cLR_2 = D_cLR(fake_img_cLR)
        D_loss_cLR_1 = mse_loss(real_d_cLR_1, 1) + mse_loss(fake_d_cLR_1, 0)
        D_loss_cLR_2 = mse_loss(real_d_cLR_2, 1) + mse_loss(fake_d_cLR_2, 0)
        D_loss = D_loss_cVAE_1 + D_loss_cLR_1 + D_loss_cVAE_2 + D_loss_cLR_2

        optim_D_cVAE.zero_grad()
        optim_D_cLR.zero_grad()
        D_loss.backward()
        optim_D_cVAE.step()
        optim_D_cLR.step()


        ''' ----------------------------- 2. Train G & E ----------------------------- '''
        # Encoded latent vector
        mu, log_variance = E(img)
        std = torch.exp(log_variance / 2)
        encoded_z = (porosity * std) + mu

        # Generate fake image and get adversarial loss
        # encoded_z = torch.cat([encoded_z[1:15].view(1, 15), flag])
        fake_img_cVAE = G(encoded_z)
        fake_d_cVAE_1, fake_d_cVAE_2 = D_cVAE(fake_img_cVAE)

        GAN_loss_cVAE_1 = mse_loss(fake_d_cVAE_1, 1)
        GAN_loss_cVAE_2 = mse_loss(fake_d_cVAE_2, 1)




        fake_img_cLR = G(porosity)
        fake_d_cLR_1, fake_d_cLR_2 = D_cLR(fake_img_cLR)

        GAN_loss_cLR_1 = mse_loss(fake_d_cLR_1, 1)
        GAN_loss_cLR_2 = mse_loss(fake_d_cLR_2, 1)

        G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2

        KL_div = lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_variance) - log_variance - 1))

        img_recon_loss = lambda_img * L1_loss(fake_img_cVAE, img)


        EG_loss = G_GAN_loss + KL_div + img_recon_loss

        all_zero_grad()
        EG_loss.backward()
        optim_E.step()
        optim_G.step()

        ''' ----------------------------- 3. Train ONLY G ----------------------------- '''

        mu_, log_variance_ = E(fake_img_cLR.detach())
        z_recon_loss = L1_loss(mu_, porosity)

        fake1 = torch.where(fake_img_cLR<0.5,torch.zeros_like(fake_img_cLR),torch.ones_like(fake_img_cLR))
        fake2 = torch.where(fake_img_cVAE<0.5,torch.zeros_like(fake_img_cVAE),torch.ones_like(fake_img_cVAE))
        pore_fake1 = 1-torch.mean(fake1,dim=-1,keepdim=True).mean(dim=-2,keepdim=True).view(-1,256)
        pore_fake2 = 1-torch.mean(fake2,dim=-1,keepdim=True).mean(dim=-2,keepdim=True).view(-1,256)

        pore = mse(pore_fake1,porosity)+mse(pore_fake2,porosity)
        G_alone_loss = lambda_z * z_recon_loss + pore

        all_zero_grad()
        G_alone_loss.backward()
        optim_G.step()
        Gloss_list.append(G_GAN_loss.item())
        Eloss_list.append(KL_div.item())
        Dloss_list.append(D_loss.item())
        imgloss_list.append(img_recon_loss.item())
        GAloss.append(G_alone_loss.item())

        print(
            '[Epoch %d/%d] [Batch %d/%d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f' \
            % (ep, epochs, i, len(dataloader), D_loss.item(), G_GAN_loss.item(), KL_div.item(), img_recon_loss.item(), G_alone_loss.item()))

    GLoss_list.append(np.mean(np.array(Gloss_list)))
    ELoss_list.append(np.mean(np.array(Eloss_list)))
    DLoss_list.append(np.mean(np.array(Dloss_list)))
    imgLoss_list.append(np.mean(np.array(imgloss_list)))
    GALoss_list.append(np.mean(np.array(GAloss)))
    if ep % 5 == 0:
        save_image(fake_img_cLR[0, 0,100].view(256, 256), 'photo/pred_%d.png' % (ep))
        save_image(img[0, 0,100].view(256, 256), 'photo/code_%d.png' % (ep))

    if ep  % 10 == 0:
        np.savetxt("loss/GLoss_%d.csv" % (ep), np.array(GLoss_list))
        np.savetxt("loss/ELoss_%d.csv" % (ep), np.array(ELoss_list))
        np.savetxt("loss/DLoss_%d.csv" % (ep), np.array(DLoss_list))
        np.savetxt("loss/imgLoss_%d.csv" % (ep), np.array(imgLoss_list))
        np.savetxt("loss/GALoss_%d.csv" % (ep), np.array(GALoss_list))

            #
        torch.save(E.state_dict(), "model/E_%d.pth" % (ep))
        torch.save(D_cLR.state_dict(), "model/D_cLR_%d.pth" % (ep))
        torch.save(D_cVAE.state_dict(), "model/D_cVAE_%d.pth" % (ep))
        torch.save(G.state_dict(), "model/G_%d.pth" % (ep))


np.savetxt("loss/GLoss.csv", np.array(GLoss_list))
np.savetxt("loss/ELoss.csv", np.array(ELoss_list))
np.savetxt("loss/DLoss.csv", np.array(DLoss_list))
np.savetxt("loss/imgLoss.csv", np.array(imgLoss_list))
np.savetxt("loss/GALoss.csv", np.array(GALoss_list))

torch.save(E.state_dict(), "model/E.pth")
torch.save(D_cLR.state_dict(), "model/D_cLR.pth" )
torch.save(D_cVAE.state_dict(), "model/D_cVAE.pth")
torch.save(G.state_dict(), "model/G.pth")
