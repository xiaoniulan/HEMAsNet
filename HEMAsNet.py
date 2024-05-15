import torch
import torch.nn as nn
import numpy as np
import math
from unit import *
# from zitrans import *
import torch.nn.functional as F
device = torch.device('cuda')

class FusionNet(nn.Module):
    #融合模块
    def __init__(self, device, n_layers, n_heads,  d_k, d_v, d_ff):
        super(FusionNet, self).__init__()
        self.encoder1 = UniversalEncoder(device, n_layers, n_heads, 228, d_k, d_v, d_ff, compute_attnv=False, dual_input=False)
        self.fc1 = nn.Linear(228,128)
        self.bn1=nn.BatchNorm1d(22)
        self.encoder2 = UniversalEncoder(device, n_layers, n_heads, 128, d_k, d_v, d_ff, compute_attnv=False, dual_input=False)
        self.fc2 = nn.Linear(128,64)
        self.elu = nn.ELU()
        self.encoder3 = UniversalEncoder(device, n_layers, n_heads, 64, d_k, d_v, d_ff, compute_attnv=False, dual_input=False)
        self.fen = nn.Sequential(
            nn.Conv1d(22,64,3,1,1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.ELU(),
            nn.Conv1d(64,128,3,1,1),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(2048,128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Linear(128,2)
        )

    def forward(self, input1):
        enc_output = self.encoder1(input1)
        enc_output = self.fc1(enc_output)
        enc_output = self.elu(self.bn1(enc_output))
        enc_output = self.encoder2(enc_output)
        enc_output = self.fc2(enc_output)
        enc_output = self.elu(self.bn1(enc_output))
        enc_output = self.fen(enc_output)

        return enc_output

class Callosum(nn.Module):
    #胼胝体模块
    def __init__(self):
        super(Callosum, self).__init__()
        self.conv1_l=nn.Conv2d(5,1,(1,5),stride=(1,5))
        self.conv1_r=nn.Conv2d(5,1,(1,5),stride=(1,5))
        self.bn = nn.BatchNorm1d(11)
        self.elu = nn.ELU()
        self.crossl=UniversalEncoder(device=device,n_layers=1, n_heads=2, d_model=100, d_k=16,d_v=16,d_ff=32, compute_attnv=True, dual_input=True).to(device)
        self.crossr=UniversalEncoder(device=device,n_layers=1, n_heads=2, d_model=100, d_k=16,d_v=16,d_ff=32, compute_attnv=True, dual_input=True).to(device)
        self.pool=nn.MaxPool1d(2)


    def forward(self, xl,xr):
        xl = self.bn(self.conv1_l(xl).squeeze())
        xr = self.bn(self.conv1_r(xr).squeeze())
        xll,attl = self.crossl(xl,xr)
        xrr,attr = self.crossr(xr,xl)
        att = torch.stack((attl,attr),dim=0)
        return xll,xrr,att

class Global(nn.Module):
    def __init__(self):
        super().__init__()
        self.Tem_l=Temporal_Block()
        self.Tem_r=Temporal_Block()
        self.Lem_l=LSTMNetwork()
        self.Lem_r=LSTMNetwork()
        self.fus = FusionNet(device=device, n_layers=1, n_heads=2, d_k=16, d_v=16, d_ff=64).to(device)
        self.bn=nn.BatchNorm1d(11)
        self.callosum=Callosum()

    def forward(self,x):

        x_l=x[:,:,0:11,:]
        x_r=x[:,:,11:22,:]
        x_l_call,x_r_call,att=self.callosum(x_l,x_r)
        x_l_cnn = self.Tem_l(x_l)
        x_r_cnn = self.Tem_r(x_r)
        x_l_lstm = self.Lem_l(x_l)
        x_r_lstm = self.Lem_r(x_r)
        x_l = torch.cat((x_l_cnn,x_l_lstm,x_l_call),2)
        x_r = torch.cat((x_r_cnn,x_r_lstm,x_r_call),2)
        x_l = self.bn(x_l)
        x_r = self.bn(x_r)
        x=torch.cat((x_l,x_r),1)
        j = self.fus(x)

        return j,att

# 创建模型实例
model = Global()
model = model.to(device)
x1 = torch.randn(56,5,22,500)
x1 = x1.to(device)
out,att = model(x1)
print(out.shape)

