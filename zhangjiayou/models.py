import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

default_config = {
    'cnn_ch_in': 1,
    'cnn_ch_out': 16,  # for now cnn_ch_out need to be the same as cnn_ch_in
    'cnn_kernel_size': 15,
    'cnn_stride': 1,
    'cnn_padding': 0,
    # output length : (ch_in - kernel + 2 * pad) / stride + 1
    # to keep length: pad = ((ch_out - 1) * stride - ch_in + kernel) / 2
    'cnn_num_layer': 4,
    'cnn_drop_prob': 0.1,
    'cnn_act_func': F.relu,
    'cnn_signal_length': 100,

    'trans_max_pos': 256,
    # 16 * (100 - 14 * 4)  # should be the same as cnn_ch_out * length
    'trans_hidden': 704,
    'trans_drop_prob': 0.1,
    'trans_num_head': 8,
    'trans_feedforward': 1024,
    'trans_num_layer': 8,
    'trans_act_func': 'relu',


    # 'cls_ch_in': 256,
    'cls_ch_out': 128,  # 88 keys + not used. probabilities for each note
    'cls_act_func': torch.sigmoid,

    'num_batch': 10,
    'num_epoch': 10,
}

debug_config = {
    'cnn_ch_in': 1,
    'cnn_ch_out': 8,  # for now cnn_ch_out need to be the same as cnn_ch_in
    'cnn_kernel_size': 15,
    'cnn_stride': 1,
    'cnn_padding': 0,
    # output length : (ch_in - kernel + 2 * pad) / stride + 1
    # to keep length: pad = ((ch_out - 1) * stride - ch_in + kernel) / 2
    'cnn_num_layer': 4,
    'cnn_drop_prob': 0.1,
    'cnn_act_func': F.relu,
    'cnn_signal_length': 100,

    'trans_max_pos': 100,
    'trans_hidden': 352,   # should be the same as cnn_ch_out
    'trans_drop_prob': 0.1,
    'trans_num_head': 8,
    'trans_feedforward': 512,
    'trans_num_layer': 8,
    'trans_act_func':  'relu',  # F.relu,

    # 'cls_ch_in': 8,
    'cls_ch_out': 128,  # 88 keys + not used. probabilities for each note
    'cls_act_func': torch.sigmoid,

    'num_batch': 10,
    'num_epoch': 10,
}


class AcousticsCNN(nn.Module):  # todo: different ch in ch out
    def __init__(self, config):
        super(AcousticsCNN, self).__init__()
        self.config = config
        self.convs = [nn.Conv1d(config['cnn_ch_in'],
                                config['cnn_ch_out'],
                                config['cnn_kernel_size'],
                                config['cnn_stride'],
                                config['cnn_padding'])]
        self.convs += [nn.Conv1d(config['cnn_ch_out'],
                                 config['cnn_ch_out'],
                                 config['cnn_kernel_size'],
                                 config['cnn_stride'],
                                 config['cnn_padding'])
                       for i in range(config['cnn_num_layer'] - 1)]
        self.convs = nn.ModuleList(self.convs)
        # self.norm = nn.LayerNorm(todo)
        # self.drop = nn.Dropout(config['cnn_drop_prob']) # will do dropout in position embedding
        self.act = config['cnn_act_func']

    def forward(self, x):
        r"""
        note that x should be shaped in [`numBatch`, `inChannel`, `signalLength`],
        and output will be shaped in [`numBatch`, `outChannel`, `signalLength'`].
        `signalLength'` is decided by `stride` and `padding`
        (default: `signalLength'` = `signalLength`).
        """
        for co in self.convs:
            x = self.act(co(x))
        # x = self.norm(x)  # todo: how many times do i need to norm
        # x = self.drop(x)
        return x


class PositionEmbedding(nn.Module):
    r"""
    adpated from transformers package by huggingface.
    """

    def __init__(self, config):
        super(PositionEmbedding, self).__init__()
        self.config = config
        self.pos_embs = nn.Embedding(config['trans_max_pos'],
                                     config['trans_hidden'])
        self.LayerNorm = nn.LayerNorm(config['trans_hidden'])
        self.dropout = nn.Dropout(config['trans_drop_prob'])

    def forward(self, input_embs):
        r"""
        `input_embs` should be shaped as [`numBatch`, `seqLength`, `hiddenSize`]
        """
        seq_length = input_embs.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_embs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_embs[:,:,0])
        position_embeddings = self.pos_embs(position_ids)

        embeddings = input_embs + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.config = config
        self.position = PositionEmbedding(config)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config['trans_hidden'],
                                                        nhead=config['trans_num_head'],
                                                        dim_feedforward=config['trans_feedforward'],
                                                        dropout=config['trans_drop_prob'],
                                                        activation=config['trans_act_func'])
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=config['trans_num_layer'])

    def forward(self, x):
        x = self.position(x)
        x = self.encoder_layer(x)
        return x


class ClassificationLayer(nn.Module):
    def __init__(self, config):
        super(ClassificationLayer, self).__init__()
        self.config = config
        self.fcn = nn.Linear(config['trans_hidden'], config['cls_ch_out'])
        self.act = config['cls_act_func']

    def forward(self, x):
        return self.act(self.fcn(x))


class MusicEye(nn.Module):
    def __init__(self, config):
        super(MusicEye, self).__init__()
        self.config = config

        self.acoustics = AcousticsCNN(config)
        self.encoder = TransformerEncoder(config)
        self.cls = ClassificationLayer(config)

        self.hidden_states = torch.zeros([config['num_batch'],
                                          config['trans_max_pos'],
                                          config['trans_hidden']]).to('cuda')
        # self.hidden_states = nn.Parameter(self.hidden_states)

    def forward(self, x):
        x = self.acoustics(x)
        x = x.view([x.shape[0], 1, -1])
        self.update_hidden_states(x)
        x = self.encoder(self.hidden_states)
        return self.cls(x)

    def update_hidden_states(self, x):
        self.hidden_states.detach_()
        self.hidden_states = torch.cat([self.hidden_states[:, 1:, :],
                                          x], dim=1)
