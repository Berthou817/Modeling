from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
import torch

data_path = os.listdir("train")
data_path.sort(key=lambda x:int(x[:-4]))

def default_loader(path):
    data_pil =  np.fromfile("train/%s"%(path),dtype=np.uint8).reshape((1,256,256,256))
    data_pil = np.where(data_pil == 1, np.ones_like(data_pil), np.zeros_like(data_pil))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor)
    return data_tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        #定义好 image 的路径
        self.images = data_path
        self.loader = loader
    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = 1-torch.mean(img,keepdim=True,dim=-1).mean(keepdim=True,dim=-2).view(-1,1)
        return img,target
    def __len__(self):
        return len(self.images)
# test=torch.zeros(8,1024,1024)







