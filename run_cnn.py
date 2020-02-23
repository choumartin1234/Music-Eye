#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models import *
import time
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from utils.spectral import spectrum
from pydub import AudioSegment
import matplotlib.pyplot as plt
import re

class SpectrumCNN(nn.Module):
    def __init__(self):
        super(SpectrumCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 8, 3, 1, 1),
            # nn.Conv2d(8, 8, 3, 1, 1),
            # nn.Conv2d(8, 8, 3, 1, 1),
            nn.Conv2d(8, 2, 3, 1, 1),
        ])
        self.pools = nn.ModuleList([
            nn.MaxPool2d(3, 2, 1),
            nn.MaxPool2d(3, 2, 1),
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1),
            nn.Dropout(0.1),
            nn.Dropout(0.1),
        ])
        self.layernorms = nn.ModuleList([
            #nn.LayerNorm([352, 161]),
            #nn.LayerNorm([176, 81]),
            #nn.LayerNorm([88, 41])
            
            #nn.LayerNorm([161, 352]),
            #nn.LayerNorm([81, 176]),
            #nn.LayerNorm([41, 88])
            nn.LayerNorm(352),
            nn.LayerNorm(176),
            nn.LayerNorm(88)
        ])

    def forward(self, x):
        c = self.convs
        p = self.pools
        d = self.dropouts
        n = self.layernorms
        relu = nn.functional.relu
        #print(x.shape)
        
        x = relu(c[0](x)) + x
        x = relu(c[1](x)) + x
        x = n[0](d[0](x))
        
        x = p[0](x)
        
        x = relu(c[2](x)) + x
        x = relu(c[3](x)) + x
        x = n[1](d[1](x))
        
        x = p[1](x)
        #print(x.shape)
        
        
        x = relu(c[4](x)) + x
        x = c[5](x)
        return torch.sigmoid(x)
#        return torch.sigmoid(c[7](c[6](x) + x))


net = SpectrumCNN()
optimizer = optim.Adam(net.parameters())
criterion = nn.BCELoss(reduction='mean')

class Mp3Dataset(Dataset):
    def __init__(self, path='data', tone_shift=True, time_shift=True, noise=True, hop_ms=3000, frame_ms=10, n_frame=50):
        super(Mp3Dataset, self).__init__()
        self.path = path
        self.tone_shift = tone_shift
        self.time_shift = time_shift
        self.noise = noise
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        
        self.mp3s = [path+'/'+i for i in os.listdir(path) if i[-4:] == '.mp3']
    
    def label_clip_to_onehot(self, label, clip_idx, clip_sec=15, frame_ms=40):
        n_frame_per_clip = clip_sec * 1000 // frame_ms
        onehot = np.zeros([2, n_frame_per_clip, 88], dtype='int') # first channel: on events, second channel: sustain.
                                                        # assume a note lasts no more than clip_sec
        frame_start = clip_idx * n_frame_per_clip
        
        for i in label:
            start = i[0] // frame_ms - frame_start
            end = (i[0] + i[2]) // frame_ms
            if end <= frame_start:
                continue
            elif start >= n_frame_per_clip:
                break
            if start >= 0:
                onehot[0, start , i[1]] = 1
            else:
                start = 0
            onehot[1, start:end, i[1]] = 1
        return onehot
    
    def __len__(self):
        return len(self.mp3s)
    
    def __getitem__(self, index):
        mp3 = self.mp3s[index]
        label = mp3.split('.')[0]+'.label'
        
        clip_idx = eval(re.sub(r"\b0*([1-9][0-9]*|0)", r"\1", mp3.split('.')[1]))
        label = [eval(i) for i in open(label)]
        label = self.label_clip_to_onehot(label, clip_idx)
        mp3 = np.array(AudioSegment.from_mp3(mp3).get_array_of_samples()) / 32768
        spec = spectrum(mp3, n_bins=352).T
        size = 1500
        if spec.shape[0] > size:
            spec = spec[:size]
        elif spec.shape[0] < size:
            spec = np.concatenate([spec, np.zeros([size-spec.shape[0], spec.shape[1]])])    
        return torch.from_numpy(spec).float(), torch.from_numpy(label).float()

class SpectrumDataset(Dataset):
    def __init__(self, path='data', tone_shift=True, time_shift=True, noise=True, hop_ms=3000, frame_ms=10, n_frame=50):
        super(SpectrumDataset, self).__init__()
        self.path = path
        self.tone_shift = tone_shift
        self.time_shift = time_shift
        self.noise = noise
        
        self.ids = [path+'/'+i[:-9] for i in os.listdir(path) if i[-9:] == '.spectrum']
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        index = self.ids[index]
        spec = index + '.spectrum'
        label = index +'.label'        
        return torch.load(spec), torch.load(label)



mididata = SpectrumDataset()

batch_size = 16
train = DataLoader(mididata, batch_size=batch_size, shuffle=True)


net.to('cuda')

import logging
logger_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("log/train.log")
handler.setFormatter(logger_format)
logger.addHandler(handler)
handler = logging.StreamHandler()
logger.addHandler(handler)

plt.figure()
plt.ion()

trainsize = len(mididata)
step_per_epoch = trainsize // batch_size
for e in range(200):
    logger.info('epoch '+str(e))    
    step = 0
    for x, y in train:
        optimizer.zero_grad()
        x, y = x.cuda(), y.cuda()
        x = x[:, :-1, :]
        x.unsqueeze_(1)
        ypred = net(x)
        #loss = criterion(ypred[:,1,:], y[:,1,:])
        loss = criterion(ypred, y) + 0.5 * e * torch.mean((1-ypred) * y)
        loss.backward()
        logger.info('epoch {} step {}/{} loss {}'.format(e,  step, step_per_epoch, loss.data.cpu().numpy()))
        optimizer.step()
        if step % (step_per_epoch // 3) == 0:
            ypreddata, ydata = ypred.data[0].cpu().numpy(), y.data[0].cpu().numpy()
            plt.title('epoch {} step {}'.format(e, step))
            plt.subplot(141)
            plt.imshow(ypreddata[0])
            plt.subplot(142)
            plt.imshow(ypreddata[1])
            plt.subplot(143)
            plt.imshow(ydata[0])
            plt.subplot(144)
            plt.imshow(ydata[1])
            plt.pause(0.2)
            plt.savefig('outputs/epoch {} step {}'.format(e, step))
        step += 1
    torch.save(net.state_dict(), 'outputs/checkpoint')
    torch.save(optimizer.state_dict(), 'outputs/optim')
plt.ioff()
plt.show()
