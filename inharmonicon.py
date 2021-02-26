# %%

# imports 
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
from colorednoise import powerlaw_psd_gaussian

# %%

# sample rate
sr = 48000

# set default sr for sounddevice
sd.default.samplerate = sr


# x values space

def oscillate(f, length=0.4, vol=1, sr=48000):
    
    # x values space, 
    # len is signal length in ms

    # how much samples to make?
    sig_length = int(sr * length)
    
    # make time window
    t = np.arange(0, length, 1/sr)

    # calculate sin signal
    sig = np.sin(2 * np.pi * t * f) * vol

    return sig


def plot_wave(sig, start=0, stop=50, sr=48000):
    
    # how much samples in 1ms?
    ms_sr = sr / 1000

    sample_start = int(start * ms_sr)
    sample_stop = int(stop * ms_sr)
    sample_length = sample_stop - sample_start

    # x axis
    x = np.linspace(
        start, 
        stop,
        sample_length)

    plt.plot(x, sig[sample_start:sample_stop])


def normalize(sig):
    max = np.max(np.abs(sig))
    return sig / max


def harmonics(f0, n=None):
    '''Return a harmonic set starting from f0.
    n - number of harmonics, if none continue to 22kHz'''
    if n:
        return [i * f0 for i in np.arange(1, n+1)]
    else:
        return [i for i in np.arange(f0, 24001, f0)]


def synthesize(freqs, length=0.4, amps=None, decay=None):
    '''Make a waveform from a list of frequencies.
    amps - a list of amplitudes for harmonics
    decay - decay constant for decay envelope'''

    # empty signal
    sig = 0

    # if empty amps, fill with ones
    if amps is None:
        amps = [1] * len(freqs)

    for f, a in zip(freqs, amps):

        sig += oscillate(f, length=length, vol=a)

    # normalize 
    norm_sig = normalize(sig)
    
    # apply decay envelope
    if decay is not None:
        # exponential decay
        d_env = np.exp(-decay * np.linspace(0, 1, len(sig)))
        norm_sig = norm_sig * d_env

    # return
    return norm_sig


def signal_to_file(sig, filename):
    sf.write(filename, sig, sr)



def tet_f(note, tuning=440):
    """Returns the frequency of a note given in scientific notation."""

    # parse note as nth semitone from A0
    letter = note[:-1]
    octave = int(note[-1])
    note_names = ["C", "C#", "D", "D#", "E",
                  "F", "F#", "G", "G#", "A", "A#", "B"]

    letter_index = note_names.index(letter) + 1
    note_number = octave * 12 + letter_index - 9

    # apply formula
    return tuning * ((2 ** (1 / 12)) ** (note_number - 49)) 


def tet_f_listwise(notes, tuning=440):
    """
    Returns frequencies of a given list of notes (ie. a chord).
    """
    out = [tet_f(note) for note in notes]
    return out



def make_rest(length):
    return np.repeat(0.0, sr * length)



def jitter(freqs, rate=.5):
    """Provide a list of jittered frequencies for a given list of frequencies."""
    i = 0
    jitters = []
    while i < len(freqs):
        if i == 0:
            # leave f0 alone
            jitters.append(freqs[0])
            i += 1
        else:
            # jitter everything else
            jitter_factor = (np.random.random(1)[0] * 2 * rate) - rate
            jitter_f = freqs[i] + (jitter_factor * freqs[0])
            # rejection sampling
            if (abs(jitter_f - freqs[i-1]) >= 30):
                jitters.append(jitter_f)
                i += 1

    return jitters



def make_jitter_rates(f0, rate=.5):
    """Provide a list of jitter rates for a given f0."""

    # if f0 is a string, convert to hertz
    if isinstance(f0, str):
        f0 = tet_f(f0)

    # make a harmonic series
    freqs = harmonics(f0)
    
    # set iterators
    i = 0
    jitter_rates = []

    # loop until all frequencies have viable jitter rates
    # rejection sampling until |f(n) - f(n-1)| > 30
    while i < len(freqs):
        if i == 0:
            # for f0 its always 1
            jitter_rates.append(1)
            i += 1
        else:
            # jitter everything else
            jitter_rate = (np.random.random(1)[0] * 2 * rate) - rate
            jitter_f = freqs[i] + (jitter_rate * freqs[0])
            # rejection sampling
            if (abs(jitter_f - freqs[i-1]) >= 30):
                jitter_rates.append(jitter_rate)
                i += 1

    return jitter_rates    


def gaussian_bp_fir_filter(sig, cutoff_f=2500):
    # try to make a gaussian bp filter

    # design a FIR filter
    t = signal.firwin(
        numtaps=3, 
        cutoff=cutoff_f, 
        window=('gaussian', 625), 
        fs=48000)

    filtered_sig = signal.filtfilt(t, a=1, x=sig)
    return filtered_sig


def peak_iir_filter(sig, freq=2500, q=3):
    # this is signal.iirpeak function
    b, a = signal.iirpeak(freq, q, sr)
    filtered_sig = signal.lfilter(b, a, sig)
    return filtered_sig


def apply_ramps(sig, ramps = 20):
    # Hann window size
    hann_size = int((ramps / 1000) * sr)
    half_hann = int(hann_size / 2)

    window = signal.windows.hann(hann_size)
    onset = window[:half_hann]
    offset = window[half_hann:]

    sig[:half_hann] *= onset
    sig[-half_hann:] *= offset
    
    return sig


def mean_f3(fs):
    """Calculate mean of third harmonics from a set of f0s"""
    x = 0
    
    for f in fs:
        x += f*3

    mean = int(x / len(fs))

    return mean


def add_pink_noise(sig, cutoff, amplitude):
    # synthesize pink noise
    noise = powerlaw_psd_gaussian(1, len(sig))

    # LPF the noise
    b, a = signal.butter(4, Wn=cutoff, fs=sr)
    noise = signal.lfilter(b, a, noise)
    noise = normalize(noise) * amplitude
    sig += noise
    sig = normalize(sig)

    # return sig with noise
    return sig



def generate_note(
    pitch, 
    type = 'h',
    jitter_rates = None,
    note_length = 400,
    decay_constant = 4.0,
    ramps = 20, 
    noise_amplitude = .05,
    noise_cutoff = None):
    """Generate a sounding note from f0/pitch.
    
    Parameters
    ----------    
    pitch : int, str 
        Pitch given in Hz or note in scientific notation.
    type: str
        `h` for harmonic, `i` for inharmonic. Harmonic is default
    jitter_rate: float
        The rate of random fluctuations in inharmonic sounds. Use None for harmonic.
    lenght: int
        Length of note in ms.
    decay_constant: float
        A decay constant for the temporal envelope. Given in s ** -1.
    ramps : int
        Size of half-Hann window amp modulation applied to onset and offset. Given in ms.
    noise_cutoff: int
        Cutoff for the noise low-pass filter
    noise_amplitude: int
        Noise amplitude.
    noise_cutoff : int
        Noise LPF cutoff in Hz.
    """
    
    # determine f0
    if isinstance(pitch, int):
        f = pitch
    else:
        f = tet_f(pitch)    

    # generate harmonics
    harmonic_fs = harmonics(f)


    if type == 'i':
        # if inharmonic tone:
        # if no rates provided, get new rates
        if jitter_rates is None:
            jitter_rates = make_jitter_rates(f)
        
        # apply rates to frequencies
        frequencies = [freq + (f * j) for freq, j in zip(harmonic_fs, jitter_rates)]
            
    else:
        # if harmonic tone:
        frequencies = harmonic_fs

    # synthesize the signal
    sig = synthesize(
        frequencies, 
        length=(note_length / 1000),
        decay=decay_constant)

    
    # apply bandpass filter
    # sig = gaussian_bp_fir_filter(sig)

    # apply iirpeak filter
    sig = peak_iir_filter(sig, q=.5)

    # if noise_cutoff, add pink noise
    if noise_cutoff is not None:
        sig = add_pink_noise(sig, noise_cutoff, noise_amplitude)
    
    
    # apply onset and offset ramps
    sig = apply_ramps(sig)
    
    return sig


def generate_chord(
    notes, 
    type='h', 
    jitter_rates=None,
    note_length = 400,
    decay_constant = 4.0,
    ramps = 20, 
    noise_amplitude = .05,
    noise_cutoff = None):
    """
    Generate chord based on pitches in scientific notation.
    """
    
    # empty freqs list
    freqs = []

    # add harmonics for every note in chord
    for note in notes:
        f0 = tet_f(note)
        harms = harmonics(f0)
        
        # if harmonic, don't jitter
        if type == 'h':
            frequencies = harms

        # if inharmonic, jitter by given jitter rate
        elif type == 'i':
            frequencies = [f + (f0 * j) for f, j in zip(harms, jitter_rates)]

        # if inharmonic, jitter
        elif type == 'ic':
            jitter_rates = make_jitter_rates(f0)
            # apply rates to frequencies
            frequencies = [f + (f0 * j) for f, j in zip(harms, jitter_rates)]

        freqs += frequencies


    # synthesize the signal
    sig = synthesize(
        freqs, 
        length=(note_length / 1000),
        decay=decay_constant)


    # apply iirpeak filter
    sig = peak_iir_filter(sig, q=.5)


    # if noise_cutoff, add pink noise
    if noise_cutoff is not None:
        sig = add_pink_noise(sig, noise_cutoff, noise_amplitude)
    
    # apply onset and offset ramps
    sig = apply_ramps(sig)

    return sig


# %%
# chords

# G#3 C3 F#4      G#3 C#4 E4    G#3 C#4 D#4    F3# B3# D4#  
# 1.	Regular   C#4 G#3 E3     c sharp minor
# 2.	Could be  C#4 A3 E3      A major
# 3.	Irregular / wrong  C4 G3 E3   C major



def make_progression(chords, type='h', jitter_rates=None, noise_cutoff=None, rest_length=.4):
    """
    Make a progression from a list of chords.
    """

    # rest length
    rest = make_rest(rest_length)

    # noise cutoff for entire progression
    if noise_cutoff is None:
        flatten = [note for chord in chords for note in chord]
        cutoff = mean_f3(tet_f_listwise(flatten))
    else:
        cutoff = noise_cutoff

    progression = []
    
    # make a chord for each chord on list
    for chord in chords:
        # if harmonic
        if type == 'h':
            ch = generate_chord(chord)
        
        # else if inharmonic
        elif type == 'i':
            ch = generate_chord(chord, type='i', jitter_rates=jitter_rates)
        
        # else if inharmonic-changing
        else:
            ch = generate_chord(chord, type='ic')
        progression.append(ch)
        progression.append(make_rest(rest_length))


    # flatten
    sig = np.concatenate(progression)

    # apply noise
    sig = add_pink_noise(sig, cutoff, .02)

    # ramp both sides
    sig = apply_ramps(sig)

    # return
    return sig


# define all chords

# ch1
ch1 = ['G#3', 'C3', 'F#4']

# ch2
ch2 = ['G#3', 'C#4', 'E4']

# ch3
ch3 = ['G#3', 'C#4', 'D#4']

# ch4
ch4 = ['F#3', 'C4', 'D#4']

# resolutions

# res1
res1 = ['C#4', 'G#3', 'E3']

# res2
res2 = ['C#4', 'A3', 'E3']

# res3
res3 = ['C4', 'G3', 'E3']

# basic progression is the same
base_prog = [ch1, ch2, ch3, ch4]

# different resolutions
prog1 = base_prog + [res1]
prog2 = base_prog + [res2]
prog3 = base_prog + [res3]

# filepath
fpath = 'audio/chords/'

# synthesize harmonic progressions
p = make_progression(prog1, type='h')
sf.write(f'{fpath}p1_h.wav', p, sr)
p = make_progression(prog2, type='h')
sf.write(f'{fpath}p2_h.wav', p, sr)
p = make_progression(prog3, type='h')
sf.write(f'{fpath}p3_h.wav', p, sr)

# inharmonic
# rate=.5
p = make_progression(prog1, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0])))
sf.write(f'{fpath}p1_i5.wav', p, sr)
p = make_progression(prog2, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0])))
sf.write(f'{fpath}p2_i5.wav', p, sr)
p = make_progression(prog3, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0])))
sf.write(f'{fpath}p3_i5.wav', p, sr)

# rate=.4
p = make_progression(prog1, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.4))
sf.write(f'{fpath}p1_i4.wav', p, sr)
p = make_progression(prog2, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.4))
sf.write(f'{fpath}p2_i4.wav', p, sr)
p = make_progression(prog3, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.4))
sf.write(f'{fpath}p3_i4.wav', p, sr)

# rate=.3
p = make_progression(prog1, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.3))
sf.write(f'{fpath}p1_i3.wav', p, sr)
p = make_progression(prog2, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.3))
sf.write(f'{fpath}p2_i3.wav', p, sr)
p = make_progression(prog3, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.3))
sf.write(f'{fpath}p3_i3.wav', p, sr)

# rate=.2
p = make_progression(prog1, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.2))
sf.write(f'{fpath}p1_i2.wav', p, sr)
p = make_progression(prog2, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.2))
sf.write(f'{fpath}p2_i2.wav', p, sr)
p = make_progression(prog3, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.2))
sf.write(f'{fpath}p3_i2.wav', p, sr)

# rate=.1
p = make_progression(prog1, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.1))
sf.write(f'{fpath}p1_i1.wav', p, sr)
p = make_progression(prog2, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.1))
sf.write(f'{fpath}p2_i1.wav', p, sr)
p = make_progression(prog3, type='i', jitter_rates=make_jitter_rates(tet_f(prog1[0][0]), rate=.1))
sf.write(f'{fpath}p3_i1.wav', p, sr)



# inharmonic changing
p = make_progression(prog1, type='ic')
sf.write(f'{fpath}p1_ic.wav', p, sr)
p = make_progression(prog2, type='ic')
sf.write(f'{fpath}p2_ic.wav', p, sr)
p = make_progression(prog3, type='ic')
sf.write(f'{fpath}p3_ic.wav', p, sr)



# %%

rest = make_rest(.1)

# ch1
notes = ['G#3', 'C3', 'F#4']
cutoff = mean_f3(tet_f_listwise(notes))
ch1 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'chord1.wav', ch1, sr)

# ch2
notes = ['G#3', 'C#4', 'E4']
cutoff = mean_f3(tet_f_listwise(notes))
ch2 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'chord2.wav', ch2, sr)

# ch3
notes = ['G#3', 'C#4', 'D#4']
cutoff = mean_f3(tet_f_listwise(notes))
ch3 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'chord3.wav', ch3, sr)

# ch4
notes = ['F#3', 'C4', 'D#4']
cutoff = mean_f3(tet_f_listwise(notes))
ch4 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'chord4.wav', ch4, sr)

# res1
notes = ['C#4', 'G#3', 'E3']
cutoff = mean_f3(tet_f_listwise(notes))
res1 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'res1.wav', res1, sr)

# res2
notes = ['C#4', 'A3', 'E3']
cutoff = mean_f3(tet_f_listwise(notes))
res2 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'res2.wav', res2, sr)

# res3
notes = ['C4', 'G3', 'E3']
cutoff = mean_f3(tet_f_listwise(notes))
res3 = generate_chord(notes, noise_cutoff=cutoff)
sf.write(fpath+'res3.wav', res3, sr)


# sequences
ch_sequence = np.concatenate((
    ch1, rest, ch2, rest, ch3, rest, ch4, rest, 
    ))
sf.write(fpath+'sequence.wav', ch_sequence, sr)



# %% 
# pentatonix

penta_a = [
    'A2', 'C3', 'D3', 'E3', 'G3', 'A3'
]

penta_b = [
    'B2', 'D3', 'E3', 'F#3', 'A3', 'B3'
]

penta_c = [
    'C3', 'D#3', 'F3', 'G3', 'A#3', 'C4'
]


mel_1 = [0, 3, 2, 1, 2]
mel_2 = [4, 2, 1, 2, 0]
mel_3 = [0, 1, 2, 1, 2]

def make_melody(steps, pentatonic):
    """
    Writes a melody in scientific notation out of given pentatonic steps.
    """
    notes = []

    # make a list of notes from list of steps
    for step in steps:
        notes.append(pentatonic[step])


    return notes


def synth_melody(notes, type='h', jitters=None, rest_length=.1):
    """
    Synthesizes a melody given a list of pitches in scientific notation or Hz.
    """
    

    # convert notes to Hz if in scientific notation
    if isinstance(notes[0], str):
        fs = [tet_f(note) for note in notes]
    else:
        fs = notes
    
    # calculate mean F3
    cutoff = mean_f3(fs)

    # empty melody set
    melody = []

    # harmonic melody
    if type == 'h':
        # synthesize each note
        for note in notes:
            sig = generate_note(note, type='h')
            melody.append(sig)
            melody.append(make_rest(rest_length))
    
    # inharmonic melody
    elif type == 'i':
        # need jitters passed as arguments
        for note in notes:
            sig = generate_note(note, type='i', jitter_rates=jitters)
            melody.append(sig)
            melody.append(make_rest(rest_length))

    # inharmonic changing
    elif type == 'ic':
        # different jitters for each note
        for note in notes:
            sig = generate_note(note, type='i')
            melody.append(sig)
            melody.append(make_rest(rest_length))
    
    # add low-passed noise
    out = np.concatenate(melody)
    out = add_pink_noise(out, cutoff, .05)

    # return melody
    return out

# %%
# write melodies and notes

def make_melodies_and_sounds(melody, pentatonic, fpath):
    # harmonic
    fname = '_a_h.wav'
    sig = synth_melody(make_melody(melody, pentatonic))
    sf.write(fpath+fname, sig, sr)

    # single notes
    for pitch in pentatonic:
        note = generate_note(pitch, type='h', noise_cutoff=tet_f(pitch)*3)
        sf.write(f'{fpath}{pitch}_h.wav', note, sr) 

    # inharmonic
    fname = '_a_i.wav'
    mel = make_melody(melody, pentatonic)
    f0 = tet_f(mel[0])
    jitt = make_jitter_rates(f0)
    sig = synth_melody(mel, jitters=jitt, type='i')
    sf.write(fpath+fname, sig, sr)

    # inharmonic single notes
    for pitch in pentatonic:
        note = generate_note(pitch, type='i', jitter_rates=jitt, noise_cutoff=tet_f(pitch)*3)
        sf.write(f'{fpath}{pitch}_i.wav', note, sr)


    # inharmonic changing
    fname = '_a_ic.wav'
    sig = synth_melody(make_melody(melody, pentatonic), type='ic')
    sf.write(fpath+fname, sig, sr)

    # single notes
    for pitch in pentatonic:
        note = generate_note(pitch, type='i', noise_cutoff=tet_f(pitch)*3)
        sf.write(f'{fpath}{pitch}_ic.wav', note, sr) 


make_melodies_and_sounds(mel_1, penta_a, 'audio/pentatonika/mel1/mel1')
make_melodies_and_sounds(mel_2, penta_a, 'audio/pentatonika/mel2/mel2')
make_melodies_and_sounds(mel_3, penta_a, 'audio/pentatonika/mel3/mel3')



# %%

# roving paradigm
# Within each stimulus train, all tones were of one frequency and were followed by a train of a different frequency. The first tone of a train was a deviant, which eventually became a standard after few repetitions. So deviants and standards have exactly the same physical properties, differing only in the number of times they have been presented. The number of times the same tone was presented varied pseudo-randomly between one and eleven. The probability that the same tone was presented once or twice was 2.5%; for three and four times the probability was 3.75% and for five to eleven times it was 12.5%. The frequency of the tones varied from 500 to 800Hz in random steps with integer multiples of 50Hz. Stimuli were presented binaurally via headphones for 15minutes. The duration of each tone was 70ms, with 5ms rise and fall times, and the inter-stimulus interval was 500ms. About 250 deviant trials (first tone presentation) were presented to each subject. About 250 deviant trials (first tone) and about 200 standards (sixth tone) were presented to each subject. Each subject adjusted the loudness of the tones to a comfortable level, which was maintained throughout the experiment


# how many reps? (Garrido et al., 2009)
# The probability that the same tone was presented once or twice was 2.5%; 
# for three and four times the probability was 3.75% and for five to eleven times it was 12.5%.
probs = ([1, 2] * 2) + ([3, 4] * 3) + ([5, 6, 7, 8, 9, 10, 11] * 10)

# CHANGE HERE
fs = list(range(500, 850, 50))

# next random freq
# randomly select number between 1 and 6
# then calculate next f index

# start with something
i = np.random.randint(0,7, 1)[0]
fs[i]

# time in seconds
time = 60


# generate list of 100 roving frequencies
rove = []
jumps = 0

while len(rove) <= 120:
    for ii in range(np.random.choice(probs, 1)[0]):
        rove.append(fs[i])
    i = (i + np.random.randint(1, 7, 1)[0]) % 7
    jumps += 1


# take list and synthesize a melody (roving)

# CHANGE HERE
fpath = 'audio/roving/500-550'

for i in range(10):
    
    # harmonic
    s = synth_melody(rove)
    sf.write(fpath+f'roving_500-800_{i}_h.wav', s, sr)

    # inharmonic 
    s = synth_melody(rove, type='i', jitters=make_jitter_rates(500))
    sf.write(fpath+f'roving_500-800_{i}_i.wav', s, sr)

    # inharmonic changing
    s = synth_melody(rove, type='ic')
    sf.write(fpath+f'roving_500-800_{i}_ic.wav', s, sr)



# same thing, different frequency list
fs = list(range(500, 570, 10))

# start with something
i = np.random.randint(0,7, 1)[0]
fs[i]

# time in seconds
time = 60


# generate list of 100 roving frequencies
rove = []
jumps = 0

while len(rove) <= 120:
    for ii in range(np.random.choice(probs, 1)[0]):
        rove.append(fs[i])
    i = (i + np.random.randint(1, 7, 1)[0]) % 7
    jumps += 1


# take list and synthesize a melody (roving)
fpath = 'audio/roving/500-560'

for i in range(10):
    
    # harmonic
    s = synth_melody(rove)
    sf.write(fpath+f'roving_500-560_{i}_h.wav', s, sr)

    # inharmonic 
    s = synth_melody(rove, type='i', jitters=make_jitter_rates(500))
    sf.write(fpath+f'roving_500-560_{i}_i.wav', s, sr)

    # inharmonic changing
    s = synth_melody(rove, type='ic')
    sf.write(fpath+f'roving_500-560_{i}_ic.wav', s, sr)



# %%
# PIOTR
def make_pair(f0_1, f0_2, type, jitters=None):
    # make rest
    rest = make_rest(.2)

    # make notes
    if jitters is not None:
        note_1 = generate_note(f0_1, type=type, jitter_rates=jitters)
        note_2 = generate_note(f0_2, type=type, jitter_rates=jitters)
    else:
        note_1 = generate_note(f0_1, type=type)
        note_2 = generate_note(f0_2, type=type)

    # glue
    sig = np.concatenate((note_1, rest, note_2))

    # add noise
    sig = add_pink_noise(sig, mean_f3([f0_1, f0_2]), amplitude=.5)

    # apply onset and offset ramps
    sig = apply_ramps(sig)
    
    return sig



diffs = [0, 2, 5, 10]

f0s = [100, 150, 200, 250, 300, 350, 400, 450]


for f0 in f0s:
    for diff in diffs:
        
        # harmonic up
        sig = make_pair(f0, f0 + diff, type='h')
        fname = f'audio/piotr/h_u_{f0}_{f0 + diff}.wav'
        sf.write(fname, sig, sr) 

        # inharmonic up
        jitters = make_jitter_rates(f0)
        sig = make_pair(f0, f0 + diff, type='i', jitters=jitters)
        fname = f'audio/piotr/i_u_{f0}_{f0 + diff}.wav'
        sf.write(fname, sig, sr) 
        
        # harmonic down
        sig = make_pair(f0, f0 - diff, type='h')
        fname = f'audio/piotr/h_d_{f0}_{f0 - diff}.wav'
        sf.write(fname, sig, sr) 

        # inharmonic down
        jitters = make_jitter_rates(f0)
        sig = make_pair(f0, f0 - diff, type='i', jitters=jitters)
        fname = f'audio/piotr/i_d_{f0}_{f0 - diff}.wav'
        sf.write(fname, sig, sr) 



## koniecznie zamień f0 na mele!

# %%



# %%

# twinkle twinkle
rest = make_rest(.2)

# frequencies
f_c = harmonics(tet_f('C4'))
f_g = harmonics(tet_f('G4'))
f_a = harmonics(tet_f('A4'))

# harmonic sounds
c = synthesize(f_c)
g = synthesize(f_g)
a = synthesize(f_a)

# inharmonic
i_c = synthesize(jitter(f_c))
i_g = synthesize(jitter(f_g))
i_a = synthesize(jitter(f_a))

# %%
sd.play(i_a)

# %%
melody = np.concatenate((c, rest, c, rest,
                  g, rest, g, rest,
                  a, rest, a, rest,
                  g))

inmelody = np.concatenate((i_c, rest, i_c, rest,
                           i_g, rest, i_g, rest,
                           i_a, rest, i_a, rest,
                           i_g))


# %%
sd.play(melody)
sf.write('audio/harmonic_twinkle.wav', melody, samplerate=sr,)
# %%
# sd.play(inmelody)
sf.write('audio/inharmonic_twinkle.wav', inmelody, samplerate=sr)

# %%

# %%


