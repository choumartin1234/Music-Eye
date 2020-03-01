import os
from utils.utils import quote
from utils.miditool import convert_to_abs_notes, transpose_tone
from utils.spectral import spectrum
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import re
from pydub import AudioSegment


def convert_midi_to_mp3(infile, outfile, bit='32k'):
    r"""
    Convert `infile` (a .mid file) to outfile (a .mp3 file) using timidity.

    Args:
        infile (str): input midi file
        outfile (str): output mp3 file
        bit (str): output bit rate. Default 32k.
    """
    os.system("timidity " + quote(infile) + " -Ow -o - |"\
        " ffmpeg -i - -acodec libmp3lame -ac 1 -ab "+bit + " " + quote(outfile))


""" this cut function's outputs are not always accurate """
# def cut_mp3(infile, segment_sec=15):
#     r"""
#     Convert `infile` (a .mp3 file) to several segment files, which are less than `segment_sec`.
#
#     Args:
#         infile (str): input midi file
#         segment_sec (int): segment size in second
#     """
#     os.system("ffmpeg -i " + quote(infile) + \
#         " -f segment -segment_time "+ str(segment_sec) +\
#             " -c copy \"" + infile[:-3] + "%02d.mp3\"")


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
        yield torch.from_numpy(s).float()    


if __name__ == '__main__':
    import logging
    logger_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("log/preprocess.log")
    handler.setFormatter(logger_format)
    logger.addHandler(handler)

    inpath = 'rawdata'
    mp3path = 'mp3data'
    spectpath = 'data'
    dic = {}
    i = 0
    for mid in os.listdir(inpath):
        try:
            mid = os.path.join(inpath, mid)
            transpose_tone(mid, 'tmp.mid', round(np.random.normal(0, 2)))
            mid = 'tmp.mid'
            out = os.path.join(mp3path, str(i))
            convert_midi_to_label(mid, out + '.label')
            convert_midi_to_mp3(mid, out + '.mp3')
            dic[i] = mid
            i += 1
        except Exception as e:
            logger.info('No.{} {}'.format(i, e))
    with open(os.path.join(mp3path, 'map'), "w") as f:
        for k, v in dic.items():
            f.write(str(k) + '\t' + str(v) + '\n')
    """ cut function is deprecated """
    # mp3s = [mp3path+'/'+i for i in os.listdir(mp3path) if i[-4:] == '.mp3']
    # for i, mp3 in enumerate(mp3s):
    #     label = mp3.split('.')[0]+'.label'
    #     for i in range(len(label)): label[i] = (label[i][0]-750, ) + label0[i][1:]
    #     clip_idx = eval(re.sub(r"\b0*([1-9][0-9]*|0)", r"\1", mp3.split('.')[1]))
    #     label = [eval(i) for i in open(label)]
    #     label = label_clip_to_onehot(label, clip_idx)
    #     mp3 = np.array(AudioSegment.from_mp3(mp3).get_array_of_samples()) / 32768
    #     spec = spectrum(mp3, n_bins=352).T
    #     size = 1500
    #     if spec.shape[0] > size:
    #         spec = spec[:size]
    #     elif spec.shape[0] < size:
    #         spec = np.concatenate([spec, np.zeros([size-spec.shape[0], spec.shape[1]])])
    #     spec, label = torch.from_numpy(spec).float(), torch.from_numpy(label).float()
    #     torch.save(spec, 'data/'+str(i)+'.spectrum')
    #     torch.save(label, 'data/'+str(i)+'.label')

    mp3s = [mp3path + '/' + i for i in os.listdir(mp3path) if i[-4:] == '.mp3']
    n_tot_clip = 0
    for mp3 in mp3s:
        label = mp3[:-4] + '.label'
        label = [eval(i) for i in open(label)]
        mp3 = np.array(AudioSegment.from_mp3(mp3).get_array_of_samples()) / 32768
        mp3 = np.concatenate([np.zeros(44100), mp3])  # add 1s offset

        len_clip = 15 * 44100
        n_clip = math.ceil(len(mp3) / len_clip)
        mp3 = np.concatenate([mp3, np.zeros(n_clip * len_clip - len(mp3))])  # pad 0

        label = [label_clip_to_onehot(label, clip, offset_ms=1000) for clip in range(n_clip)]
        spec = [spectrum(mp3[clip * len_clip:(clip + 1) * len_clip], n_bins=352) for clip in range(n_clip)]
        for s, l in zip(spec, label):
            # plt.figure(figsize=(16,16))
            # plt.subplot(131)
            # plt.imshow(s)
            # plt.subplot(132)
            # plt.imshow(l[0])
            # plt.subplot(133)
            # plt.imshow(l[1])
            # plt.imshow()
            path = os.path.join(spectpath, str(n_tot_clip))
            torch.save(torch.from_numpy(s).float(), path+'.spectrum')
            torch.save(torch.from_numpy(l).float(), path+'.label')
            n_tot_clip += 1
