from models import *
import torch
import time

config = default_config#debug_config

net = MusicEye(debug_config).to('cuda')


t0 = time.time()
for step in range(100):
    x = torch.rand([config['num_batch'], config['cnn_ch_in'], config['cnn_signal_length']])
    x = x.to('cuda')
    # print(net(x))
    net(x)
t1 = time.time()

print(int((t1 - t0) * 1000))
print(net(x))
