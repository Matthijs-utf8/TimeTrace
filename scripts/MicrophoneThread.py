import numpy as np
import pyaudio
import threading
from queue import Queue

class AudioProcessor:
    """
    An audio processor class for capturing and processing audio data.

    Args:
        audio_queue (Queue): A queue to store captured audio data.
        toggle_whisper (threading.Event): An event to control audio playback.
        chunk_size (int): The size of audio chunks to capture.
        fs (int): The sampling frequency of the audio.

    Attributes:
        audio_queue (Queue): A queue to store captured audio data.
        toggle_whisper (threading.Event): An event to control audio playback.
        chunk_size (int): The size of audio chunks to capture.
        fs (int): The sampling frequency of the audio.
    """

    def __init__(self, audio_queue: Queue, toggle_whisper: threading.Event, chunk_size: int, fs: int):
        self.audio_queue = audio_queue
        self.toggle_whisper = toggle_whisper
        self.chunk_size = chunk_size
        self.fs = fs

    def fetch_signal(self):
        """
        Start capturing audio data and store it in the audio queue.

        This method opens an audio stream and continuously captures audio data,
        which is then put into the audio_queue. It also plays back the audio if
        the toggle_whisper event is set.
        """
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            output=True,
            frames_per_buffer=self.chunk_size
        )
        try:
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=True)
                audio_data = np.frombuffer(data, dtype=np.int16)
                if self.toggle_whisper.is_set() and stream.get_write_available() > 0:  # Check if the speaker is on
                    stream.write(audio_data.tobytes())
                self.audio_queue.put(audio_data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
