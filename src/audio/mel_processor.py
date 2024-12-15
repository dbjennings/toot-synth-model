import numpy as np
import librosa

# Default audio processing parameters
DEFAULT_N_FFT: int = 2048  # STFT window size
DEFAULT_HOP_LENGTH: int = 512  # Number of samples between successive frames
DEFAULT_SAMPLE_RATE: int = 22050


class MelProcessor:
    """
    A class for processing audio signals and converting them to different spectral representation.
    Implements efficient mel-spectrogram conversion with caching and normalization.
    """

    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = 128,
        min_db: float = -100.0,
        ref_db: float = 20.0,
        center: bool = True,
        power: float = 2.0,
    ):
        """
        Initialize the MelProcessor with customizable processing parameters.

        Args:
            n_fft: Size of the FFT window (samples)
            hop_length: Number of samples between successive FFT windows
            n_mels: Number of mel frequency bins
            min_db: Minimum decibel threshold for normalization
            ref_db: Reference decibel level for normalization
            center: Whether to pad signal at the beginning and end
            power: Exponent for the magnitude spectrogram
        """

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.min_db = min_db
        self.ref_db = ref_db
        self.center = center
        self.power = power

        # Caching matrices for efficiency
        self._mel_basis = None
        self._inv_mel_basis = None

    def _get_mel_basis(self, sample_rate: int) -> np.ndarray:
        """
        Lazily compute and cache the mel filterbank matrix.

        Args:
            sample_rate: Audio sample rate in Hz

        Returns:
            np.ndarray: Mel filterbank matrix
        """

        if self._mel_basis == None:
            self._mel_basis = librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=self.n_mels
            )
        return self._mel_basis

    def _get_inv_mel_basis(self, sample_rate: int) -> np.ndarray:
        """
        Lazily compute and cache the inverse mel filterbank matrix.

        Args:
            sample_rate: Audio sample rate in Hz

        Returns:
            np.ndarray: Inverse mel filterbank matrix
        """
        if self._inv_mel_basis == None:
            self._inv_mel_basis = np.linalg.pinv(self._get_mel_basis(sample_rate))
        return self._inv_mel_basis

    def normalize(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the spectrogram to a decibel scale within [0, 1].

        Args:
            stft_matrix: Input spectrogram

        Returns:
            np.ndarray: Normalized spectrogram
        """

        return np.clip(
            (20 * np.log10(np.maximum(1e-5, stft_matrix)) - self.min_db)
            / (self.ref_db - self.min_db),
            0,
            1,
        )

    def denormalize(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Convert normalized spectrogram back to linear scale

        Args:
            stft_matrix: Input STFT spectrogram

        Returns:
            np.ndarray: Denormalized spectrogram
        """

        return np.power(
            10.0, (stft_matrix * (self.ref_db - self.min_db) + self.min_db) / 20.0
        )

    def stft_to_mel(
        self,
        stft_matrix: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Convert STFT spectrogram to mel spectrogram.

        Args:
            stft_matrix: Input STFT spectrogram
            sample_rate: Audio sample rate in Hz
            normalize: Whether to normalize the output

        Returns:
            np.ndarray: Mel spectrogram

        Raises:
            ValueError: If input matrix is invalid
        """

        if not isinstance(stft_matrix, np.ndarray):
            raise ValueError("stft_matrix must be a numpy matrix")

        if stft_matrix.size == 0:
            raise ValueError("stft_matrix is empty")

        mel_matrix = np.dot(
            self._get_mel_basis(sample_rate), np.abs(stft_matrix) ** self.power
        )

        return self.normalize(mel_matrix) if normalize else mel_matrix

    def audio_to_mel(
        self,
        raw_audio: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        normalize: bool = True,
        trim_silence: bool = False,
    ) -> np.ndarray:
        """
        Convert raw audio to mel spectrogram.

        Args:
            raw_audio: Input audio time series
            sample_rate: Audio sample rate in Hz
            normalize: Whether to normalize the output
            trim_silence: Whether to remove silence from the audio

        Returns:
            np.ndarray: Mel spectrogram

        Raises:
            ValueError: If input audio is invalid
            RuntimeError: If STFT computation fails
        """

        if not isinstance(raw_audio, np.ndarray):
            raise ValueError("raw_audio must be a numpy array")

        if raw_audio.size < self.n_fft:
            raise ValueError(
                f"raw_audio ({raw_audio.size} samples) must be larger than n_fft {self.n_fft}"
            )

        # Optionally trim silence for cleaner processing
        if trim_silence:
            raw_audio, _ = librosa.effects.trim(raw_audio, top_db=30)

        try:
            stft_matrix = librosa.stft(
                raw_audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=self.center,
            )
        except Exception as e:
            raise RuntimeError(f"STFT computation failed: {str(e)}")

        return self.stft_to_mel(
            stft_matrix, sample_rate=sample_rate, normalize=normalize
        )

    def mel_to_stft(
        self,
        mel_matrix: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        denormalize: bool = True,
    ):
        """
        Convert mel spectrogram back to STFT spectrogram.

        Args:
            mel_matrix: Input mel spectrogram
            sample_rate: Audio sample rate in Hz

        Returns:
            np.ndarray: STFT spectrogram

        Raises:
            ValueError: If input matrix is invalid
        """

        if not isinstance(mel_matrix, np.ndarray):
            raise ValueError("mel_matrix must be a numpy array")

        if mel_matrix.size == 0:
            raise ValueError("mel_matrix is empty")

        if denormalize:
            mel_matrix = self.denormalize(mel_matrix)

        stft_matrix = np.dot(self._get_inv_mel_basis(sample_rate), mel_matrix)
        stft_matrix = np.maximum(1e-5, stft_matrix)

        return stft_matrix

    def mel_to_audio(
        self,
        mel_matrix: np.ndarray,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        denormalize: bool = True,
        n_iter: int = 32,
    ):
        """
        Convert mel spectrogram back to audio using Griffin-Lim algorithm.

        Args:
            mel_matrix: Input mel spectrogram
            sample_rate: Audio sample rate in Hz
            denormalize: Whether to denormalize the input
            n_iter: Number of Griffin-Lim iterations

        Returns:
            np.ndarray: Reconstructed audio time series

        Raises:
            ValueError: If input matrix is invalid
        """

        if not isinstance(mel_matrix, np.ndarray):
            raise ValueError("mel_matrix must be a numpy array")

        if mel_matrix.size == 0:
            raise ValueError("mel_matrix is empty")

        stft_matrix = self.mel_to_stft(
            mel_matrix, sample_rate=sample_rate, denormalize=denormalize
        )

        audio = librosa.griffinlim(
            stft_matrix,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            center=self.center,
            n_iter=n_iter,
        )

        return audio
