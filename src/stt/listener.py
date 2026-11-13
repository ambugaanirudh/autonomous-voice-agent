"""Orchestrates continuous microphone capture, VAD, wake word, and transcription.

The Listener records audio in 30 ms frames, runs WebRTC VAD on each frame,
buffers speech segments, transcribes them when silence is detected, and
extracts commands after the wake word.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from src.stt.transcriber import Transcriber
from src.stt.vad import VAD
from src.stt.wake_word import WakeWordDetector


class Listener:
    def __init__(
        self,
        wake_word: str = "computer",
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        vad_aggressiveness: int = 2,
        silence_timeout_sec: float = 1.5,
        max_utterance_sec: float = 10.0,
        model_size: str = "small",
        language: str = "en",
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.silence_timeout_sec = silence_timeout_sec
        self.max_utterance_sec = max_utterance_sec

        self.vad = VAD(
            aggressiveness=vad_aggressiveness,
            sample_rate=sample_rate,
            frame_duration_ms=frame_duration_ms,
        )
        self.wake_word = WakeWordDetector(wake_word=wake_word)
        self.transcriber = Transcriber(
            model_size=model_size, language=language, device="cpu", compute_type="int8"
        )

        # Derived values
        self.frame_size = self.vad.frame_size
        self.frames_per_second = int(1000 / frame_duration_ms)
        self.silence_frames = int(silence_timeout_sec * self.frames_per_second)
        self.max_frames = int(max_utterance_sec * self.frames_per_second)

    def listen_once(self, timeout_sec: float = 30.0) -> Optional[str]:
        """Listen for a wake word + command, return the command text or None.

        Blocks until:
          - A command is detected (wake word + speech followed by silence), or
          - timeout_sec elapses with no speech.

        Returns:
            The command text after the wake word, or None on timeout/noise.
        """
        print(f"Listening (wake word: '{self.wake_word.wake_word}')...")
        audio_buffer: list[np.ndarray] = []
        silent_frames = 0
        speech_started = False
        total_frames = 0

        def callback(
            indata: np.ndarray, frames: int, timestamp_info, status
        ) -> None:
            nonlocal silent_frames, speech_started, total_frames, audio_buffer
            if status:
                # Input overflow etc. — log but keep going
                pass

            # Convert float32 (-1..1) to int16 bytes for VAD
            mono = indata[:, 0] if indata.ndim > 1 else indata
            pcm_bytes = self.vad.float_to_pcm16(mono)

            try:
                is_speech = self.vad.is_speech(pcm_bytes)
            except ValueError:
                # Frame size mismatch — skip this frame
                return

            total_frames += 1

            if is_speech:
                speech_started = True
                silent_frames = 0
                audio_buffer.append(mono.copy())
            else:
                if speech_started:
                    silent_frames += 1

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            callback=callback,
        ):
            # Wait for either a complete utterance or timeout
            import time

            start = time.time()
            while time.time() - start < timeout_sec:
                if speech_started and silent_frames >= self.silence_frames:
                    break
                if total_frames >= self.max_frames:
                    # Utterance too long — cut it off
                    break
                sd.sleep(100)

        if not audio_buffer:
            return None

        # Concatenate buffered audio and transcribe
        full_audio = np.concatenate(audio_buffer)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, full_audio, self.sample_rate)
            text = self.transcriber.transcribe(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not text:
            return None

        if not self.wake_word.contains_wake_word(text):
            print(f"Heard (no wake word): {text}")
            return None

        command = self.wake_word.extract_command(text)
        if command:
            print(f"Command heard: {command}")
        return command