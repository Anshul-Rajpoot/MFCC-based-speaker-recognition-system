import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from sklearn.metrics.pairwise import cosine_similarity


class MFCCProcessor:
    def __init__(
        self,
        sr=16000,
        frame_size=25,
        frame_stride=10,
        n_fft=512,
        n_mels=20,
        n_mfcc=13,
        pre_emphasis=0.97
    ):
        """
        MFCC Processor with advanced feature support
        """
        self.sr = sr
        self.frame_size_ms = frame_size
        self.frame_stride_ms = frame_stride
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.pre_emphasis_coeff = pre_emphasis

    # --------------------------------------------------
    #               SIGNAL PROCESSING
    # --------------------------------------------------
    def pre_emphasize(self, signal):
        return np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1])

    def frame_signal(self, signal, sr):
        frame_length = int(round(self.frame_size_ms * sr / 1000))
        frame_step = int(round(self.frame_stride_ms * sr / 1000))

        signal_length = len(signal)
        num_frames = int(np.ceil(abs(signal_length - frame_length) / frame_step)) + 1

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros(pad_signal_length - signal_length)
        pad_signal = np.append(signal, z)

        indices = (
            np.tile(np.arange(0, frame_length), (num_frames, 1))
            + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        )

        frames = pad_signal[indices.astype(np.int32)]
        return frames

    def apply_window(self, frames):
        return frames * np.hamming(frames.shape[1])

    def compute_power_spectrum(self, frames):
        mag_frames = np.abs(np.fft.rfft(frames, self.n_fft))
        return (1.0 / self.n_fft) * (mag_frames ** 2)

    def create_mel_filterbank(self, sr):
        return librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.n_mels)

    def apply_mel_filterbank(self, pow_frames, mel_fbanks):
        mel_energy = np.dot(pow_frames, mel_fbanks.T)
        return np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)

    def compute_mfcc(self, log_mel_energy):
        mfcc = dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        return mfcc

    # --------------------------------------------------
    #           ADVANCED FEATURE EXTRACTION
    # --------------------------------------------------
    def compute_deltas(self, mfcc):
        delta = librosa.feature.delta(mfcc.T).T
        delta2 = librosa.feature.delta(mfcc.T, order=2).T
        return delta, delta2

    def compute_energy(self, frames):
        return np.sum(frames ** 2, axis=1)

    def compute_zcr(self, frames):
        return np.mean(librosa.feature.zero_crossing_rate(frames.T), axis=1)

    def summarize_features(self, feature_matrix):
        """
        Convert variable-length features to fixed-size vector
        """
        return np.concatenate([
            np.mean(feature_matrix, axis=0),
            np.std(feature_matrix, axis=0)
        ])

    # --------------------------------------------------
    #               FULL PIPELINE
    # --------------------------------------------------
    def full_pipeline(self, signal, sr):
        emphasized = self.pre_emphasize(signal)
        frames = self.frame_signal(emphasized, sr)
        windowed = self.apply_window(frames)
        power_spectrum = self.compute_power_spectrum(windowed)

        mel_fbanks = self.create_mel_filterbank(sr)
        mel_energy = self.apply_mel_filterbank(power_spectrum, mel_fbanks)
        log_mel_energy = np.log(mel_energy)

        mfcc = self.compute_mfcc(log_mel_energy)
        delta, delta2 = self.compute_deltas(mfcc)

        energy = self.compute_energy(frames)
        zcr = self.compute_zcr(frames)

        # ML-ready feature vector
        feature_vector = np.concatenate([
            self.summarize_features(mfcc),
            self.summarize_features(delta),
            self.summarize_features(delta2),
            [np.mean(energy), np.std(energy)],
            [np.mean(zcr), np.std(zcr)]
        ])

        return {
            "mfcc": mfcc,
            "delta": delta,
            "delta2": delta2,
            "energy": energy,
            "zcr": zcr,
            "feature_vector": feature_vector,
            "mel_fbanks": mel_fbanks
        }

    # --------------------------------------------------
    #           SPEAKER SIMILARITY
    # --------------------------------------------------
    @staticmethod
    def speaker_similarity(vec1, vec2):
        """
        Cosine similarity between two speaker feature vectors
        """
        return cosine_similarity(
            vec1.reshape(1, -1),
            vec2.reshape(1, -1)
        )[0][0]

    # --------------------------------------------------
    #               PLOTTING UTILITIES
    # --------------------------------------------------
    @staticmethod
    def plot_mfcc(mfcc, sr, title="MFCC Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfcc.T, sr=sr, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("MFCC Coefficients")
        return fig

    @staticmethod
    def plot_feature_variance(mfcc):
        fig, ax = plt.subplots(figsize=(8, 4))
        variance = np.var(mfcc, axis=0)
        ax.bar(range(len(variance)), variance)
        ax.set_title("MFCC Variance (Feature Importance Proxy)")
        ax.set_xlabel("MFCC Index")
        ax.set_ylabel("Variance")
        return fig



    # --------------------------------------------------
    #           BASIC VISUALIZATION UTILITIES
    # --------------------------------------------------
    @staticmethod
    def plot_time_domain(signal, sr, title="Time Domain Signal"):
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(signal, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return fig

    @staticmethod
    def plot_spectrogram(signal, sr, title="Spectrogram"):
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal)), ref=np.max
        )
        img = librosa.display.specshow(
            D, sr=sr, x_axis="time", y_axis="log", ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(title)
        return fig
