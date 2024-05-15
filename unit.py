import torch
import torch.nn as nn
import numpy as np
import math
device = torch.device('cuda')


class CustomScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads, compute_attnv=False):
        super(CustomScaledDotProductAttention, self).__init__()
        self.d_k = d_k  # 每个头部的维度
        self.n_heads = n_heads  # 头部数量
        self.compute_attnv = compute_attnv  # 是否计算头部的平均注意力

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]  # 查询向量
        K: [batch_size, n_heads, len_k, d_k]  # 键向量
        V: [batch_size, n_heads, len_v(=len_k), d_v]  # 值向量
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # 计算注意力分数
        attn = nn.Softmax(dim=-1)(scores)  # 应用softmax函数获取注意力权重
        context = torch.matmul(attn, V)  # 计算上下文向量

        if self.compute_attnv:
            attnv = torch.mean(attn, dim=1)  # 计算所有头部的平均注意力
            return context, attnv
        else:
            return context


class MultiHeadAttention(nn.Module):
    def __init__(self, device, n_heads, d_model, d_k, d_v, compute_attnv=False):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.device = device
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.compute_attnv = compute_attnv

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attention_module = CustomScaledDotProductAttention(self.d_k, self.n_heads, self.compute_attnv)
        result = attention_module(Q, K, V)
        # Check if the attention vector is also returned
        if self.compute_attnv:
            context, attnv = result
        else:
            context = result

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.layer_norm(output + residual)

        if self.compute_attnv:
            return output, attnv
        else:
            return output


class UniversalEncoderLayer(nn.Module):
    def __init__(self, device, n_heads, d_model, d_k, d_v, d_ff, compute_attnv=False, dual_input=False):
        super(UniversalEncoderLayer, self).__init__()
        self.compute_attnv = compute_attnv
        self.dual_input = dual_input
        self.enc_self_attn = MultiHeadAttention(device, n_heads, d_model, d_k, d_v, compute_attnv=compute_attnv)
        self.pos_ffn = PoswiseFeedForwardNet(device, d_model, d_ff)

    def forward(self, inputs1, inputs2=None):
        if self.dual_input and inputs2 is not None:
            outputs = self.enc_self_attn(inputs1, inputs2, inputs2)
            if self.compute_attnv:
                outputs, attn = outputs
            else:
                outputs = outputs
        else:
            outputs = self.enc_self_attn(inputs1, inputs1, inputs1)
            if self.compute_attnv:
                outputs, attn = outputs
            else:
                outputs = outputs
        outputs = self.pos_ffn(outputs)
        if self.compute_attnv:
            return outputs, attn
        else:
            return outputs

class UniversalEncoder(nn.Module):
    def __init__(self, device, n_layers, n_heads, d_model, d_k, d_v, d_ff, compute_attnv=False, dual_input=False):
        super(UniversalEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [UniversalEncoderLayer(device, n_heads, d_model, d_k, d_v, d_ff, compute_attnv, dual_input) for _ in
             range(n_layers)])

    def forward(self, inputs1, inputs2=None):
        outputs = inputs1

        for layer in self.layers:
            if self.layers[0].compute_attnv:
                outputs, attn = layer(outputs, inputs2 if self.layers[0].dual_input else None)
            else:
                outputs = layer(outputs, inputs2 if self.layers[0].dual_input else None)

        if self.layers[0].compute_attnv:
            return outputs, attn
        else:
            return outputs

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,device,d_model,d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual) # [batch_size, seq_len, d_model]


class down_sample(nn.Module):
    def __init__(self, inc, kernel_size, stride, padding):
        super(down_sample, self).__init__()
        self.conv = nn.Conv2d(in_channels = inc, out_channels = inc, kernel_size = (1, kernel_size), stride = (1, stride), padding = (0, padding), bias = False)
        self.bn = nn.BatchNorm2d(inc)
        self.elu = nn.ELU(inplace = False)

    def forward(self, x):
        output = self.elu(self.bn(self.conv(x)))
        return output


class input_layer(nn.Module):
    def __init__(self, outc, groups):
        super(input_layer, self).__init__()
        self.conv_input = nn.Conv2d(in_channels = 1, out_channels = outc, kernel_size = (1, 3),
                                    stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn_input = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)

    def forward(self, x):
        output = self.bn_input(self.conv_input(x))
        return output

class Residual_Block(nn.Module):
    def __init__(self, inc, outc, groups = 1):
        super(Residual_Block, self).__init__()
        if inc is not outc:
          self.conv_expand = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = 1,
                                       stride = 1, padding = 0, groups = groups, bias = False)
        else:
          self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels = inc, out_channels = outc, kernel_size = (1, 3),
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = (1, 3),
                               stride = 1, padding = (0, 1), groups = groups, bias = False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.elu = nn.ELU(inplace = False)


    def forward(self, x):
        if self.conv_expand is not None:
          identity_data = self.conv_expand(x)
        else:
          identity_data = x
        output = self.bn1(self.conv1(x))
        output = self.bn2(self.conv2(output))
        output = torch.add(output,identity_data)
        return output

def embedding_network(input_block, Residual_Block, num_of_layer, outc, groups = 1):
    layers = []
    layers.append(input_block(outc,groups=groups))
    for i in range(0, num_of_layer):
        layers.append(Residual_Block(inc = int(math.pow(2, i)*outc), outc = int(math.pow(2, i+1)*outc),
                                     groups = groups))
    return nn.Sequential(*layers)


class Multi_Scale_Temporal_Block(nn.Module):
    #多尺度卷积
    def __init__(self, outc, num_of_layer = 1):
        super().__init__()
        self.num_of_layer = num_of_layer
        self.embedding = embedding_network(input_layer, Residual_Block, num_of_layer = num_of_layer, outc = outc, groups=1)
        self.downsampled1 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 4, 4, 0)
        self.downsampled2 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 8, 8, 0)
        self.downsampled3 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 16, 16, 0)
        self.downsampled4 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)
        self.downsampled5 = down_sample(outc*int(math.pow(2, num_of_layer))+1, 32, 32, 0)

    def forward(self, x):
        embedding_x = self.embedding(x)#([128, 8, 15, 1280])
        cat_x = torch.cat((embedding_x, x), 1)#([128, 5, 22, 1280])

        downsample1 = self.downsampled1(cat_x)
        downsample2 = self.downsampled2(cat_x)
        downsample3 = self.downsampled3(cat_x)
        downsample4 = self.downsampled4(cat_x)
        downsample5 = self.downsampled5(cat_x)
        temporal_fe = torch.concat((downsample1,downsample2,downsample3,downsample4,downsample5),3)

        return temporal_fe

class Temporal_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mstblock1 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock2 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock3 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock4 = Multi_Scale_Temporal_Block(outc=2)
        self.mstblock5 = Multi_Scale_Temporal_Block(outc=2)

        self.conv = nn.Sequential(
            nn.Conv2d(25, 64, kernel_size=(1,7),stride=(1,2)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64,128,(1,5),1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128,1,1),
        )
        self.ch = nn.Sequential(
            nn.Linear(117,64)
        )



    def forward(self,x):

        t_fe1 = self.mstblock1(x[:,0,:,:].unsqueeze(1))
        t_fe2 = self.mstblock2(x[:,1,:,:].unsqueeze(1))
        t_fe3 = self.mstblock3(x[:,2,:,:].unsqueeze(1))
        t_fe4 = self.mstblock4(x[:,3,:,:].unsqueeze(1))
        t_fe5 = self.mstblock5(x[:,4,:,:].unsqueeze(1))
        t_fe = torch.cat((t_fe1,t_fe2,t_fe3,t_fe4,t_fe5),1)
        t_fe = self.conv(t_fe).squeeze()
        t_fe = self.ch(t_fe)

        return t_fe


class LSTMM(nn.Module):
    def __init__(self):
        super(LSTMM, self).__init__()
        self.in_fc = nn.Linear(500,128)
        self.conv = nn.Conv1d(11,64,1)
        self.bn = nn.BatchNorm1d(64)

        self.lstm1 = nn.LSTM(input_size=64,hidden_size=16,num_layers=2,batch_first=True,bidirectional=True,dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=32,hidden_size=8,num_layers=2,batch_first=True,bidirectional=True,dropout=0.2)
        self.lstm3 = nn.LSTM(input_size=16,hidden_size=4,num_layers=1,batch_first=True,bidirectional=True)
        self.pool = nn.MaxPool1d(2)


    def forward(self, x):

        x = self.bn(self.conv(x)).transpose(1,2)
        x,_ = self.lstm1(x)
        x = self.pool(x.transpose(1,2)).transpose(1,2)
        x,_ = self.lstm2(x)
        x = self.pool(x.transpose(1,2)).transpose(1,2)
        x = self.lstm3(x)[0].transpose(1,2)

        return x


class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork, self).__init__()

        self.ff1=LSTMM()
        self.ff2=LSTMM()
        self.ff3=LSTMM()
        self.ff4=LSTMM()
        self.ff5=LSTMM()
        self.conv = nn.Sequential(
            nn.Conv1d(40,11,1),
            nn.ELU(),
            nn.Linear(125,64)
        )

    def forward(self, x):

        f_fe1 = self.ff1(x[:,0,:,:])
        f_fe2 = self.ff2(x[:,1,:,:])
        f_fe3 = self.ff3(x[:,2,:,:])
        f_fe4 = self.ff4(x[:,3,:,:])
        f_fe5 = self.ff5(x[:,4,:,:])
        f_fe = torch.cat((f_fe1,f_fe2,f_fe3,f_fe4,f_fe5),1)
        f_fe = self.conv(f_fe)

        return f_fe
