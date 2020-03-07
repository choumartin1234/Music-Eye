import os
import midi
from .utils import quote
from collections import namedtuple
import numpy as np
import math
import random

def play(file):
    r"""
    Use timidity play midi file
    """
    os.system("timidity " + quote(file))


def transpose_tone(infile, outfile, bias, change_tempo=False):
    r"""
    Transpose the tone of a midi file. If `bias` is negative,
    it becomes flat. If `bias` is  positive, it becomes sharp.

    Args:
        infile (str): input midi file
        outfile (str): output midi file
        bias (int): the distance the tone shifts. If the bias is too large,
            and makes some note out of [21, 108], the bias will be modified.
        change_tempo (bool): if randomly change tempo (by setting resolution)
    """

    pattern = midi.read_midifile(infile)

    if pattern.format not in (0, 1):
        raise ValueError(
            "Pattern format is not 0 or 1. Format 2 is not supported.")

    m, M = -128, 128
    for track in pattern:
        for evt in track:
            if isinstance(evt, (midi.NoteOnEvent, midi.NoteOffEvent)):
                m = max(21 - evt.data[0], m)
                M = min(108 - evt.data[0], M)
    bias = min(max(m, bias), M)
    for track in pattern:
        for evt in track:
            if isinstance(evt, (midi.NoteOnEvent, midi.NoteOffEvent)):
                evt.data[0] += bias
    if change_tempo: pattern.resolution = math.ceil(2**(np.log2(pattern.resolution) + np.random.rand()*2) - 1)
    midi.write_midifile(outfile, pattern)


def restrict_instrument(infile, outfile, allowset=None,
                        forbidset=[], reassign=False, remove_drum=True):
    r"""
    Restrict the midi instrument in `allowset` for every track.
    If `allowset` is `None`, the `allowset` will be the instruments
    not in `forbidset`. The subsitution for forbidden instruments is piano.
    
    Args:
        infile (str): input midi file
        outfile (str): output midi file
        allowset (list): a list of ids of allowed instruments
        forbidset (list): a list of ids of forbidden instruments
    """
    pattern = midi.read_midifile(infile)
    pattern.make_ticks_abs()

    if allowset is None:
        allowset = [i for i in range(128) if i not in forbidset]

    if pattern.format not in (0, 1):
        raise ValueError(
            "Pattern format is not 0 or 1. Format 2 is not supported.")

    new_tracks = []
    drum_channel = [9]
    for track in pattern:
        new_track = []
        for evt in track:
            if isinstance(evt, (midi.NoteOnEvent, midi.NoteOffEvent)):
                if (evt.channel in drum_channel) and remove_drum:
                    continue
            if isinstance(evt, midi.ProgramChangeEvent):
                if evt.data[0] >= 113 and remove_drum:
                    drum_channel.append(evt.channel)
                    continue
                if evt.data[0] == 14:
                    evt.data[0] = 0
                if reassign or evt.data[0] not in allowset:
                    evt.data[0] = random.choice(allowset)
            new_track.append(evt)
        new_track = midi.Track(new_track, tick_relative=False)
        new_tracks.append(new_track)
    
    if len(new_tracks) == 0: raise ValueError('empty pattern')
    pattern = midi.Pattern(new_tracks, resolution=pattern.resolution, format=pattern.format, tick_relative=False)
    pattern.make_ticks_rel()
    midi.write_midifile(outfile, pattern)

class AbsNote():
    r"""
    Records absolute time, pitch, and abosolute duration.
    """

    # from dangen/MidiToText
    def __init__(self, time, pitch, dur, instru='piano'):
        r"""
        Args:
            time: absolute time the note was played (microsecond)
            pitch: pitch
            dur: absolute duration of the note (microsecond)
            instru: instrument in ('piano', 'violin', 'cello', 'guitar', 'flute', 'other')
        """
        self.time = time
        self.pitch = pitch
        self.dur = dur
        instrus = ('piano', 'violin', 'cello', 'guitar', 'flute', 'other')
        self.instru = instru if instru in instrus else 'other'

    def __str__(self):
        return str((self.time, self.pitch, self.dur))


# yapf: disable
def simplifyMidiEvent(evt):
    r"""
    Simplify 27 MIDI events into 5 and make them more readable.
    Returns tick, simplified data, event type.
    """
    if isinstance(evt, midi.NoteOnEvent):
        if evt.data[1] == 0:
            return evt.tick, evt.data[0], 'off'
        else:
            return evt.tick, evt.data[0], 'on'
    elif isinstance(evt, midi.NoteOffEvent):
        return evt.tick, evt.data[0],    'off'
    elif isinstance(evt, midi.EndOfTrackEvent):
        return evt.tick, 0,              'end'
    elif isinstance(evt, midi.SetTempoEvent):
        return evt.tick, evt.get_mpqn(), 'tempo'
    else:
        return evt.tick, 0,              'unk'
# yapf: enable


def get_simplified_event_list(file):
    r"""
    Parse `file` (.mid) and return simplified events, containing only NoteOnEvent,
    NoteOffEvent, SetTempoEvent, EndOfTrackEvent.
    Returns simplified events, resolution.
    """
    pattern = midi.read_midifile(file)
    pattern.make_ticks_abs()
    resolution = pattern.resolution  # resolution is a const

    if pattern.format not in (0, 1):
        raise ValueError(
            "Pattern format is not 0 or 1. Format 2 is not supported.")

    def parse_track(track):
        events = []
        for evt in track:
            evt = simplifyMidiEvent(evt)
            if evt[-1] != 'unk':
                events.append(evt)
        return events

    if pattern.format == 0:
        tracks = [parse_track(pattern[0])]
    elif pattern.format == 1:
        tracks = [parse_track(i) for i in pattern]
        tracks = [tracks[0] + tracks[i] for i in range(1, len(tracks))]

    # sort by absolute tick. note that sorted is stable
    tracks = [sorted(i, key=lambda x: (x[0], x[1:])) for i in tracks]
    return tracks, resolution


def convert_to_abs_notes(file):
    r"""
    Returns a list of `AbsNote`, which contains abosulte time, pitch,
    and absolute duration.
    """
    tracks, resolution = get_simplified_event_list(file)

    notes = []

    for track in tracks:
        num_on = sum(i[-1] == 'on' for i in track)
        num_off = sum(i[-1] == 'off' for i in track)
        if num_on != num_off:
            raise ValueError("Corrupted MIDI. NoteOnEvent count = {}, "
                             "NoteOffEvent count = {}".format(num_on, num_off))

        # microsecond per tick = 6e7 / bpm / resolution.
        us_per_tick = 6e7 / 120 / resolution  # default bpm = 120
        ons = {}  # pitch -> absolute time list
        abstime = 0
        abstick = 0

        def found_off(pitch):
            if pitch not in ons:
                return
            on = ons[pitch].pop(0)
            if len(ons[pitch]) == 0:
                ons.pop(pitch)
            duration = abstime - on
            note = AbsNote(on, pitch, duration)
            notes.append(note)

        for evt in track:
            abstime += (evt[0] - abstick) * us_per_tick
            abstick = evt[0]
            if evt[-1] == 'tempo':
                # us_per_tick = mpqn / resolusion
                us_per_tick = evt[1] / resolution
            elif evt[-1] == 'on':
                # if evt[1] in ons: found_off(evt[1]); print('on overlaps!')
                # if evt[1] in ons:
                #     raise ValueError("Corrupted MIDI. NoteOnEvents overlapped.")
                if evt[1] in ons:
                    ons[evt[1]].append(abstime)
                else:
                    ons[evt[1]] = [abstime]
            elif evt[-1] == "off":
                found_off(evt[1])

        # # some ons may not be closed
        if len(ons) != 0:
            raise ValueError("Corrupted MIDI. Unclosed NoteOnEvent found.")
        # keys = list(ons.keys())
        # for pitch in keys:  found_off(pitch) print('off missing!')
    return sorted(notes, key=lambda x: (x.time, ))
