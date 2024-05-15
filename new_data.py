from torch.utils.data import Dataset,DataLoader
from torch import tensor
import os
import scipy.io as sio
import numpy as np

class si_dataset_mt(Dataset):

    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.img_path=os.listdir(self.root_dir)
        self.name2label = {'c':0,'d':1}
        self.data_list = list()
        for file in os.listdir(self.root_dir):
            self.data_list.append(os.path.join(self.root_dir, file))
        print("Load {} Data Successfully!".format(type))

    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, item1):
        si_item = self.data_list[item1]
        name = '{}'.format(si_item[44:])
        lable = self.name2label[os.path.basename(si_item)[0][0]]
        si_item = sio.loadmat('{}'.format(si_item))
        sig_tm = si_item['content']
        sig_tm=tensor(sig_tm)
        lable = tensor(lable)
        sig_tm=sig_tm.float()
        sig_tm= np.array(sig_tm[:,:, :])
        sig_tm=tensor(sig_tm)#.float()
        sig_tm=sig_tm.transpose(0,1)
        lable=lable.float()


        return sig_tm,lable,name

