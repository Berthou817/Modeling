from dataloader  import *
from dataloader import trainset
from bicyclegan import *
import torch
import numpy as np
if not os.path.exists("result"):
    os.mkdir("result")
else:
    os.rmdir("result")
    os.mkdir("result")
G = torch.nn.DataParallel(StyleGenerator()).cuda()
mod = torch.load('doc1/model/G.pth')
G.load_state_dict(mod)
G.eval()
train_data  = trainset()
dataloader = DataLoader(train_data, batch_size=1)
flag = 0
for im,j in dataloader:
    with torch.no_grad():
        res = G(j.cuda().view(1,256))
        flag+=1
        print("%d"%(flag))
        res = torch.where(res < 0.5, torch.zeros_like(res), torch.ones_like(res))
    res = res.data.cpu().numpy()
    res.astype(np.uint8).tofile("result/res_%d.raw"%(flag))

