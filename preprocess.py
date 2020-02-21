import os
from utils.utils import quote
from utils.miditool import convert_to_abs_notes


def convert_midi_to_mp3(infile, outfile, bit='64k'):
    r"""
    Convert `infile` (a .mid file) to outfile (a .mp3 file) using timidity.

    Args:
        infile (str): input midi file
        outfile (str): output mp3 file
        bit (str): output bit rate. Default 64k.
    """
    os.system("timidity " + quote(infile) + " -Ow -o - |"\
        " /usr/local/ffmpeg/bin/ffmpeg -i - -acodec libmp3lame -ab "+bit + " " + quote(outfile))


def convert_midi_to_label(infile, outfile):
    label = convert_to_abs_notes(infile)
    with open(outfile, "w") as f:
        for i in label:
            # seems that timidity always drops the opening silence
            # if your tool does keep the silence, remove the following line
            i[0] -= label[0][0] 
            f.write(str(i) + '\n')


if __name__ == '__main__':
    import logging
    logger_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("log/preprocess.log")
    handler.setFormatter(logger_format)
    logger.addHandler(handler)

    path = 'data'
    dic = {}
    i = 0
    for mid in ['albinoni1.mid','Cras numquam scire.mid']:#os.listdir(path):
        try:
            mid = os.path.join(path, mid)
            out = os.path.join(path, str(i))
            convert_midi_to_label(mid, out + '.label')
            convert_midi_to_mp3(mid, out + '.mp3')
            dic[i] = mid
            i += 1
        except Exception as e:
            logger.info('No.{} {}'.format(i, e))
    with open(os.path.join(path, 'map'), "w") as f:
        for k, v in dic.items():
            f.write(str(k) + '\t' + str(v) + '\n')
