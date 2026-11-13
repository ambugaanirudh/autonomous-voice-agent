"""Local speech-to-text using faster-whisper (INT8, CPU by default)."""

from __future__ import annotations

from faster_whisper import WhisperModel


class Transcriber:
    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        self.language = language
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to a WAV/FLAC/MP3 file (16 kHz mono preferred).

        Returns:
            Transcribed text, stripped of leading/trailing whitespace.
        """
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            vad_filter=True,  # faster-whisper's built-in Silero VAD pre-filter
        )
        text = " ".join(segment.text for segment in segments)
        return text.strip()