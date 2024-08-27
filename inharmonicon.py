import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf
import pyloudnorm as pyln


class Harmonics:
    def __init__(
            self,
            f0: float,
            jitter_rate: float = 0,
            jitter_factors: np.ndarray = None,
            fmin: int = 1,
            fmax: int = None,
            fs=48000,
    ):
        """Create a harmonic series from a given fundamental frequency (f0).

        Args:
            f0 (float): Fundamental frequency to start the harmonic series
            jitter_rate (float, optional): Apply jitter to make the series inharmonic. Set to 0 for no jittering. Defaults to 0.
            jitter_factors (np.ndarray, optional): Use pre-made jitter factors. These can be taken from other Harmonics objects with a "get_factors()" method. If None, generate new jitter factors. Defaults to None.
            fmin (int, optional): Number of the starting harmonic. Use 1 to start with f0. Defaults to 1.
            fmax (int, optional): Number of the final harmonic in the series. If None, generate harmonics up to the Nyquist limit. Defaults to 100.
            fs (int, optional): Project sample rate. Defaults to 48000.
        """
        # set nyquist lymit
        self.nyquist = fs / 2

        # set f0
        self.f0 = f0

        # set jitter
        self.jitter_rate = jitter_rate

        self.harmonic_series = self.generate_harmonic_series(f0, fmin, fmax)

        if jitter_factors is None:
            self.series, self.factors = self.jitter()
        else:
            self.factors = jitter_factors
            self.series = self.apply_jitter_factors(jitter_factors)

    def __str__(self):
        """Convienience method for printing out Harmonics objects."""
        return np.array_str(self.series)

    def __repr__(self):
        """Convienience method for printing out Harmonics objects."""
        return np.array_str(self.series)

    def plot(self, lower_limit=None, upper_limit=None):
        """Plot the frequencies in the series against the harmonic series.

        Args:
            lower_limit (float, optional): Lower limit for frequency. If None, plot everything. Defaults to None.
            upper_limit (float, optional): Upper limit for frequency. If None, plot everything. Defaults to None.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        if lower_limit is None:
            lower_limit = 0
        if upper_limit is None:
            upper_limit = self.nyquist

        df = pd.DataFrame({"harm": self.harmonic_series, "jittered": self.series}).melt(
            value_name="frequency", var_name="sound"
        )
        fig, ax = plt.subplots()
        sns.stripplot(data=df, x="sound", y="frequency", jitter=False, marker=5, ax=ax)
        ax.set_ylim(lower_limit, upper_limit)
        plt.show()

    def generate_harmonic_series(
            self, f0: float, fmin: int = 1, fmax: int = None
    ) -> np.ndarray:
        """Return a harmonic series starting at f0.

        Args:
            f0 (float): Fundamental frequency
            fmin (int, optional): Which harmonic to start with. Defaults to 1 (fundamental frequency).
            fmax (int, optional): Which harmonic is the final one in the series. If None, make harmonics up to the Nyquist limit. Defaults to None.

        Returns:
            np.ndarray: Ndarray containing the harmonic series.
        """
        if fmax is None:  # go up to Nyquist
            return np.arange(f0 * fmin, self.nyquist, f0)
        else:
            count = np.arange(
                fmin, fmax + 1
            )  # fmax+1 because np.arange is right-exclusive
            return count * f0

    def get_series(self) -> np.ndarray:
        """Return the (in)harmonic series.

        Returns:
            np.ndarray: The (in)harmonic series.
        """
        return self.series

    def get_factors(self) -> np.ndarray:
        """Return the jitter factors.

        Returns:
            np.ndarray: The jitter factors.
        """
        return self.factors

    def jitter(self) -> tuple[np.ndarray, np.ndarray]:
        """Jitter the harmonic series to generate inharmonicity.

        Returns:
            (harmonic_series, factors): Harmonic series, jitter factors
        """

        if self.jitter_rate == 0:
            factors = np.zeros(len(self.harmonic_series))
            return self.harmonic_series, factors
        else:
            factors = []
            jittered_harmonics = []
            for i, harmonic in enumerate(self.harmonic_series):
                # always leave f0 alone
                if harmonic == self.f0:
                    jittered_harmonics.append(harmonic)
                    factors.append(0)
                else:
                    # rejection sampling
                    # check if previous harmonic is not 30Hz or closer
                    while True:
                        jfactor = np.random.default_rng().uniform(
                            -self.jitter_rate, self.jitter_rate
                        )
                        current = harmonic + harmonic * jfactor
                        if len(jittered_harmonics) > 0:
                            prev = jittered_harmonics[-1]
                        else:
                            prev = 0
                        if abs(current - prev) > 30:
                            jittered_harmonics.append(current)
                            factors.append(jfactor)
                            break
            jittered_harmonics = np.array(jittered_harmonics)
            factors = np.array(factors)

            return jittered_harmonics, factors

    def apply_jitter_factors(self, jitter_factors):
        res = self.harmonic_series + (self.harmonic_series * jitter_factors)
        return res


class Sound:
    def __init__(
            self,
            f: float | np.ndarray | Harmonics,
            length: float,
            amp: float = 1.0,
            fs: int = 48000,
    ):
        """Make a sound.

        Args:
            f (float): Frequency or an array of frequencies to synthesize.
            length (float): Sound length in seconds.
            amp (float, optional): Sound max amplitude. Defaults to 1.0.
            fs (int, optional): Sample rate. Defaults to 48000.
        """
        # set sample rate
        self.fs = fs

        # set length (in seconds)
        self.length = length

        # set time vector
        self.t = np.arange(0, self.length, 1 / self.fs)

        # set length in samples
        self.length_s = len(self.t)

        # if f is integer, make a pure tone
        if isinstance(f, int):
            self.sound = self.generate_pure_tone(f, amp)
            self.normalize()

        # else if f is an ndarray, make a complex tone
        elif isinstance(f, np.ndarray):
            self.generate_complex_tone(f, phases="random")
            self.apply_tapers()

        elif isinstance(f, Harmonics):
            self.generate_complex_tone(f.series, phases="random")
            self.apply_tapers()

    def generate_pure_tone(
            self, f: float, amp: float = 1.0, phase: float = 0
    ) -> np.ndarray:
        """Synthesize a pure tone.

        Args:
            f (float): Frequency.
            amp (float, optional): Amplitude. Defaults to 1.0.
            phase (float, optional): Phase in radians. Defaults to 0.

        Returns:
            np.ndarray: An ndarray that holds the tone.
        """
        # calculate pure tone
        tone = amp * np.sin(2 * np.pi * f * self.t + phase)
        return tone

    def play(self):
        """Play the sound using default system audio interface.

        This method uses sounddevice. Use only for interactive sessions and for small scripts. It cannot be used for multiple overlapping playbacks.
        """
        sd.play(self.sound, self.fs)

    def plot(self, start=0, stop=0.05):
        import matplotlib.pyplot as plt
        """Plot a segment of the waveform.

        Args:
            start (int, optional): Start time in seconds. Defaults to 0.
            stop (float, optional): End time in seconds. Defaults to 0.05.
        """
        sample_start = int(start * self.fs)
        sample_stop = int(stop * self.fs)
        sample_length = sample_stop - sample_start

        # x axis
        x = np.linspace(start, stop, sample_length)

        # plot
        plt.plot(x, self.sound[sample_start:sample_stop], linewidth=0.5, c="tab:red")
        plt.show()

    def normalize(self, scale=1):
        """Normalize the signal to full-scale (max amplitude = 1). Operates in place."""
        # max = np.max(np.abs(self.sound))
        # self.sound = self.sound / max
        # self.sound = self.sound * scale
        # measure the loudness first 
        meter = pyln.Meter(self.fs)  # create BS.1770 meter
        loudness = meter.integrated_loudness(np.tile(self.sound, 10))

        # # loudness normalize audio to -12 dB LUFS
        self.sound = pyln.normalize.loudness(self.sound, loudness, -12.0)

    def generate_complex_tone(
            self,
            harmonics: npt.ArrayLike,
            amps: npt.ArrayLike = None,
            phases: npt.ArrayLike | int | str = 0,
    ):
        """Generate a complex tone.

        Args:
            harmonics (npt.ArrayLike): Array with frequencies of consecutive harmonics.
            amps (npt.ArrayLike, optional): Array with amplitudes of consecutive harmonics. Must be the same shape as `harmonics`. If `None`, all amplitudes are full-scale (1.0). Defaults to None.
            phases (npt.ArrayLike, int or str, optional): Array with phases of consecutive harmonics. Must be the same shape as `harmonics`. If `random`, all harmonics are in random phase. If 0, all harmonics are in zero-phase. Defaults to 0.

        Returns:
            np.ndarray: Array with the complex tone.
        """
        # initialize
        res = np.zeros(self.length_s)

        # cut frequencies so that they don't go above nyquist limit
        harmonics = self.cut_to_nyquist(harmonics)

        # if default, set all amplitudes to 1
        if amps is None:
            amps = np.ones(self.length_s)

        # if default, set all phases to 0
        if phases is None:
            phases = np.zeros(self.length_s)
        # else if 'random', make random phase:
        elif phases == "random":
            phases = np.random.default_rng().uniform(0, 2 * np.pi, self.length_s)

        # add up harmonics to get the final complex tone
        for harmonic, amp, phase in zip(harmonics, amps, phases):
            res += self.generate_pure_tone(harmonic, amp, phase)

        self.sound = res

        # normalize or you'll blow up your speakers
        self.normalize()

    def apply_tapers(self, taper_length: float = 0.005):
        """Apply a Hanning taper to the sound.

        Args:
            taper_length (float, optional): Length of the onset/offset. Defaults to .005.
        """
        half_hann = int(self.fs * taper_length)
        window = np.hanning(2 * half_hann)
        onset = window[:half_hann]
        offset = window[half_hann:]

        self.sound[:half_hann] *= onset
        self.sound[-half_hann:] *= offset

    def cut_to_nyquist(self, frequencies: np.ndarray) -> np.ndarray:
        """Removes frequencies above the Nyquist limit from the frequencies array.

        Args:
            frequencies (np.ndarray): Frequencies to check.

        Returns:
            np.ndarray: Ndarray containing frequencies below the Nyquist limit.
        """
        nyquist = self.fs / 2
        return frequencies[frequencies < nyquist]

    def filter(self, freq, order: int = 4, type="hp"):
        """Filter the sound inplace with scipy's Butterworth filter.

        Args:
            freq (int, float, list): Cutoff frequency. For band-pass filter, supply a list of two frequencies.
            order (int, optional): Filter order. Defaults to 4.
            type (str, optional): Filter type. Use 'hp' for high-pass, 'lp' for low-pass and 'bp' for band-pass. Defaults to "hp".
        """
        from scipy.signal import sosfilt, butter

        # design a filter
        sos = butter(order, freq, type, fs=self.fs, output="sos")
        # apply
        self.sound = sosfilt(sos, self.sound)

    def adjust_volume(self, volume: float):
        pass

    def save(self, fname: str):
        """Save sound under a given filename. Uses soundfile under the hood.

        Args:
            fname (str): Filename with *.wav extension.
        """
        sf.write(file=fname, data=self.sound, samplerate=self.fs)
