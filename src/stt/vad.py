"""Voice Activity Detection using WebRTC VAD.

Operates on 16 kHz mono int16 PCM audio. VAD aggressiveness 0-3:
  0 = least aggressive (more false positives)
  3 = most aggressive (more false negatives)
"""

from __future__ import annotations

import webrtcvad
import numpy as np


class VAD:
    def __init__(
        self,
        aggressiveness: int = 2,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
    ) -> None:
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("frame_duration_ms must be 10, 20, or 30")
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

    def is_speech(self, pcm16_bytes: bytes) -> bool:
        """Return True if the given 16-bit PCM frame contains speech.

        Args:
            pcm16_bytes: Raw int16 PCM bytes. Length must match frame_size * 2 bytes.
        """
        if len(pcm16_bytes) != self.frame_size * 2:
            raise ValueError(
                f"Expected {self.frame_size * 2} bytes for a "
                f"{self.frame_duration_ms}ms frame at {self.sample_rate}Hz, "
                f"got {len(pcm16_bytes)}"
            )
        return self.vad.is_speech(pcm16_bytes, self.sample_rate)

    def float_to_pcm16(self, audio: np.ndarray) -> bytes:
        """Convert float32 numpy array (-1..1) to int16 PCM bytes."""
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        return pcm16.tobytes()