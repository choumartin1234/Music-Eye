#%%
import os
from utils.utils import quote


def convert_midi_to_mp3(infile, outfile):
    r"""
    Convert `infile` (a .mid file) to outfile (a .mp3 file) using timidity.

    Args:
        infile (str): input midi file
        outfile (str): output midi file
    """
    os.system("timidity " + quote(infile) + " -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k " + quote(outfile))


import music21
music21.stream

# %%
