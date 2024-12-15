import numpy as np
import librosa
from typing import Tuple

VALID_WINDOWS = [
    "hann",
    "hamming",
    "blackman",
    "barthann",
    "bohman",
    "boxcar",
    "cosine",
    "tukey",
    "triang",
]

VALID_PAD_MODES = ["reflect", "constant", "edge", "linear_ramp"]


class AudioSTFTProcessor:
    """
    A processing class for the conversions between audio time-series and complex Short Time Fourier Transform (STFT) matrices
    """

    def __init__(
        self, n_fft=2048, hop_length=512, window="hann", center=True, pad_mode="reflect"
    ):

        # Input validation
        if not isinstance(n_fft, int) or n_fft <= 0:
            raise ValueError("n_fft must be a positive integer")

        if not isinstance(hop_length, int) or hop_length <= 0:
            raise ValueError("hop_length must be a positive integer")

        if window not in VALID_WINDOWS:
            raise ValueError(f"window must be one of {VALID_WINDOWS}")

        if pad_mode not in VALID_PAD_MODES:
            raise ValueError(f"pad_mode must be one of {VALID_PAD_MODES}")

        # Store validated parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def process_audio(
        self, raw_audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts raw audio data to its STFT representation

        Args:
            raw_audio: raw audio times series data (1D array)

        Returns:
            tuple: (complex STFT matrix, magnitudes, phases)
                - STFT matrix shape: [1 + n_fft/2, n_frames]
                - Magnitudes are absolute values of the STFT
                - Phases are the angles of the complex STFT values
        """

        # Input Validation
        if not isinstance(raw_audio, np.ndarray):
            raise ValueError("raw_audio must be a numpy array")

        if raw_audio.ndim != 1:
            raise ValueError("raw_audio must be a 1-dimensional array")

        if len(raw_audio) < self.n_fft:
            raise ValueError(
                f"Audio sample length ({len(raw_audio)}) must be more than n_fft ({self.n_fft})"
            )

        if not np.isfinite(raw_audio).all():
            raise ValueError("raw_audio contains invalid values")

        # Compute the STFT
        try:
            stft_matrix = librosa.stft(
                raw_audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                center=self.center,
                pad_mode=self.pad_mode,
            )

            # Calculate magnitudes and phases
            magnitudes = np.abs(stft_matrix)
            phases = np.angle(stft_matrix)

            return stft_matrix, magnitudes, phases

        except Exception as e:
            raise RuntimeError(f"STFT computation failed: {str(e)}")

    def inverse_transform(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Reconstructs the audio signal from its STFT representation

        Args:
            stft_matrix: Complex STFT matrix

        Returns:
            np.ndarray: Reconstructed audio time-series
        """

        # Input validation
        if not isinstance(stft_matrix, np.ndarray):
            raise ValueError("stft_matrix must be a numpy array")

        if stft_matrix.ndim != 2:
            raise ValueError("stft_matrix must be a 2-dimensional array")

        if not np.iscomplex(stft_matrix):
            raise ValueError("stft_matrix must be complex-valued")

        try:
            return librosa.istft(
                stft_matrix=stft_matrix,
                hop_length=self.hop_length,
                window=self.window,
                center=self.center,
            )
        except Exception as e:
            raise RuntimeError("Inverse STFT computation failed: ")

    def get_time_points(self, num_samples: int, sample_rate: int) -> np.ndarray:
        """
        Calculate the time points corresponding to each STFT frame

        Args:
            num_samples: calculated as (sample rate) * (duration)
            sample_rate: in Hz

        Returns:
            Array of time points in frames
        """
        # Input validation
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        if num_samples < self.n_fft:
            raise ValueError(
                f"num_samples ({num_samples}) must be more than n_fft ({self.n_fft})"
            )

        return librosa.frames_to_time(
            np.arange(1 + (num_samples - self.n_fft) // self.hop_length),
            sr=sample_rate,
            hop_length=self.hop_length,
        )


if __name__ == "__main__":
    pass
