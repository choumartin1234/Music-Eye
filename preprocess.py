import os
from utils.utils import quote
from utils.miditool import convert_to_abs_notes, transpose_tone, restrict_instrument
from utils.spectral import spectrum
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import re
from pydub import AudioSegment
import traceback

import imageio
import multiprocessing


def convert_midi_to_mp3(infile, outfile, bit='32k'):
    r"""
    Convert `infile` (a .mid file) to outfile (a .mp3 file) using timidity.

    Args:
        infile (str): input midi file
        outfile (str): output mp3 file
        bit (str): output bit rate. Default 32k.
    """
    assert 0 == os.system("timidity " + quote(infile) + " -Ow -o - |"\
        " ffmpeg -y -i - -acodec libmp3lame -ac 1 -ab "+bit + " " + quote(outfile))


def float_to_uint8(s):
    s = ((s - np.log2(0.01)) / (1 - np.log2(0.01)) * 255)
    return s.astype(np.uint8)

def uint8_to_float(s):
    return  np.log2(0.01) + s * (1 - np.log2(0.01)) / 255


def convert_midi_to_label(infile, outfile):
    label = convert_to_abs_notes(infile)
    with open(outfile, "w") as f:
        offset = label[0].time
        for i in label:
            # seems that timidity always drops the opening silence
            # if your tool does keep the silence, remove the following line
            i.time -= offset
            i.time = int(i.time / 1000)
            i.pitch -= 21
            if i.pitch < 0 or i.pitch > 108: continue
            i.dur = int(i.dur / 1000)
            f.write(str(i) + '\n')


def label_clip_to_onehot(label, clip_idx, clip_sec=15, frame_ms=40, offset_ms=0):
    n_frame_per_clip = clip_sec * 1000 // frame_ms
    onehot = np.zeros([2, n_frame_per_clip, 88], dtype='int')  # first channel: on events, second channel: sustain.
    # assume a note lasts no more than clip_sec
    frame_start = clip_idx * n_frame_per_clip - offset_ms // frame_ms

    for i in label:
        start = i[0] // frame_ms - frame_start
        end = (i[0] + i[2]) // frame_ms - frame_start
        if end <= 0:
            continue
        elif start >= n_frame_per_clip:
            break
        if start >= 0:
            onehot[0, start, i[1]] = 1
        else:
            start = 0
        onehot[1, start:end, i[1]] = 1
    return onehot


def clip_and_spectrum(mp3):
    len_clip = 15 * 44100
    n_clip = math.ceil(len(mp3) / len_clip)
    mp3 = np.concatenate([mp3, np.zeros(n_clip * len_clip - len(mp3))])  # pad 0

    for clip in range(n_clip):
        s = spectrum(mp3[clip * len_clip:(clip + 1) * len_clip], n_bins=352)
        #s = uint8_to_float(float_to_uint8(s))
        yield torch.from_numpy(s).float()    


if __name__ == '__main__':
    import logging
    logger_format = logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("log/preprocess.log")
    handler.setFormatter(logger_format)
    logger.addHandler(handler)

    inpath = 'rawdata'
    mp3path = 'mp3data'
    spectpath = 'data'
    mp3dict = {}
    num_mp3 = 0

    lr = lambda l, r: list(range(l, r))
    basics = lr(0, 4),  lr(16, 20), lr(4, 12), lr(24, 26), lr(39, 42), lr(70, 77)
    others = lr(16, 20)+[21]+[46]+lr(50,54)+lr(56, 61)+[63, 68] + [79, 85, 88, 98, 102, 103, 105, 106, 107]
    weighted_instrus = others[:] # copy
    for i in basics: 
        weighted_instrus += math.ceil(len(others) / len(i)) * i



    midi_files = [os.path.join(inpath, mid) for mid in os.listdir(inpath)]
    np.random.shuffle(midi_files)
    for mid in midi_files:
        try:
            for i in range(4):
                transpose_tone(mid, 'tmp.mid', round(np.random.normal(0, 12))) # 2 to 12
                if i == 0:
                    restrict_instrument('tmp.mid', 'tmp.mid') # remove drum
                elif i == 1:
                    restrict_instrument('tmp.mid', 'tmp.mid', basics[0], reassign=True)
                else:
                    restrict_instrument('tmp.mid', 'tmp.mid', weighted_instrus, reassign=True)
                out = os.path.join(mp3path, str(num_mp3))
                convert_midi_to_label('tmp.mid', out + '.label')
                convert_midi_to_mp3('tmp.mid', out + '.mp3')
                mp3dict[num_mp3] = mid
                num_mp3 += 1

        except Exception as e:
            logger.info('{} {}: {}'.format(mid, e, traceback.format_exc()))
    with open(os.path.join(mp3path, 'map'), "w") as f:
        for k, v in mp3dict.items():
            f.write(str(k) + '\t' + str(v) + '\n')
    
    
    specdict = {}
    mp3s = [mp3path + '/' + i for i in os.listdir(mp3path) if i[-4:] == '.mp3']
    n_tot_clip = 0
    for mp3 in mp3s:
        label = mp3[:-4] + '.label'
        label = [eval(i) for i in open(label)]
        offset_ms = np.random.randint(1000, 4000)
        mp3 = np.array(AudioSegment.from_mp3(mp3).get_array_of_samples()) / 32768
        mp3 = np.concatenate([np.zeros(int(44.1 * offset_ms)), mp3])  # add offset

        len_clip = 15 * 44100
        n_clip = math.ceil(len(mp3) / len_clip)
        mp3 = np.concatenate([mp3, np.zeros(n_clip * len_clip - len(mp3))])  # pad 0

        label = [label_clip_to_onehot(label, clip, offset_ms=offset_ms) for clip in range(n_clip)]
        spec = [spectrum(mp3[clip * len_clip:(clip + 1) * len_clip], n_bins=352) for clip in range(n_clip)]
        for s, l in zip(spec, label):
            path = os.path.join(spectpath, str(n_tot_clip))
            
            s = float_to_uint8(s)
            imageio.imwrite(path+'.jpg', s.astype(np.uint8))
            imageio.imwrite(path+'.0.jpg', l[0])
            imageio.imwrite(path+'.1.jpg', l[1])
            # torch.save(torch.BoolTensor(l), path+'.label')
            
            specdict[n_tot_clip] = mp3
            n_tot_clip += 1
    
    with open(os.path.join(spectpath, 'map'), "w") as f:
        for k, v in specdict.items():
            f.write(str(k) + '\t' + str(v) + '\n')
                              
