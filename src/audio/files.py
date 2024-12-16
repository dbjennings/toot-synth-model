from pathlib import Path
from typing import Optional, Tuple, Dict, List
import librosa
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from .mel import MelProcessor


@dataclass
class ProcessorConfig:
    n_fft: int = 2048  # STFT window size
    hop_length: int = 512  # Sample step size between iterations
    n_mels: int = 128  # Mel spectrogram window size
    sample_rate: int = 22050  # Default sample rate
    trim_silence: bool = True  # Remove silence from audio ends
    normalize: bool = True  # Enable normalization across samples
    source_rel_path: Path = Path("assets/raw_audio")
    sink_rel_path: Path = Path("assets/mels")


class FileProcessor:
    """
    A processor class that batch converts .wav files to their Mel spectrogram representations.
    Attempts automatic project root discovery if none is provided.
    """

    def __init__(self, project_root: str, config: Optional[ProcessorConfig]):
        self.project_root = self._get_project_root(project_root)
        self.source_path = self.project_root / self.config.source_rel_path
        self.sink_path = self.project_root / self.config.sink_rel_path
        self.sink_path.mkdir(parents=True, exist_ok=True)

        self.config = config | ProcessorConfig()

        self.mel_processor = MelProcessor(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )

        self.stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
        }

    def process(self) -> Tuple[Dict, List[Dict]]:
        """
        Processes all .wav files present in the source directory.
        Outputs Mel spectrograms in .npy format to sink directory.

        Returns:
            Tuple containing
                - Overall project file processing statistics
                - A list of individual file processing statistics
        """
        file_stats = []

        # Get all .wav files in source directory
        audio_files = list(self.source_path.rglob("*.wav"))

        # Check for audio files
        if not audio_files:
            return self.stats, file_stats

        # Iterate through audio files
        for audio_file in tqdm(audio_files, desc="Processing audio files:"):
            file_stat = self._process_single_file(audio_file)
            file_stats.append(file_stat)

        return self.stats, file_stats

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Loads and standardizes audio files

        Args:
        Returns:
            audio_data: Mono audio data with standard sample rate
        """
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_path)

        # Convert to mono audio if necessary
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample audio if necessary
        if sample_rate != self.config.sample_rate:
            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=self.config.sample_rate
            )

        return audio_data

    def _process_single_file(self, audio_path: Path) -> Optional[Dict]:
        """
        Loads a .wav, processes it, and saves its Mel spectrogram

        Args:
            audio_path: path to the .wav file
        Returns:
            An (optional) dictionary with file processing statistics
        """
        try:
            # Generate output path
            output_name = audio_path.name.split(".")[0]
            output_path = self.sink_path / output_name / ".npy"

            # Skip file if already processed
            if output_path.exists():
                self.stats["skipped"] += 1
                return None

            # Load and process audio
            audio_data = self._load_audio(audio_path)

            # Convert to Mel spectrogram representation
            mel_spec = self.mel_processor.audio_to_mel(
                audio_data,
                sample_rate=self.config.sample_rate,
                normalize=self.config.normalize,
                trim_silence=self.config.trim_silence,
            )

            # Save the processed data
            np.save(output_path, mel_spec)

            # Update stats
            self.stats["processed"] += 1

            # Return processing statistics
            return {
                "input_path": audio_path,
                "output_path": output_path,
                "shape": mel_spec.shape,
            }

        except Exception as e:
            self.stats["failed"] += 1
            return None

    def _get_project_root(self, project_root: Optional[str | Path] = None) -> Path:
        """
        Resolves the project root directory.
        Attempts to walk down the current directory if no project root is provided.

        Args:
            project_root: path to the root directory of the project
        Returns:
            root: the resolved project root directory
        Raises:
            FileNotFoundError: no project root found in automated attempts
        """
        if project_root is not None:
            root = Path(project_root).resolve()

        # Attempt automatic project_root retrieval
        else:
            current_path = Path(__file__).resolve()
            root = None

            # Iterate top-down from the current directory
            for path in current_path.parents:
                if (path / "src").exists():
                    root = path
                    break

            # No root found
            if root is None:
                raise FileNotFoundError(
                    "Could not automatically detect project root."
                    "Please provide project_root parameter"
                )

        return root


if __name__ == "__main__":
    file_processer = FileProcessor()

    process_stats = file_processer.process()
