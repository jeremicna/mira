import threading
import time
import av
import numpy as np
from collections import deque

class AudioTap:
    def __init__(self, url, sample_rate=48000, channels=2, buffer_seconds=30):
        """
        url: RTSP URL from MediaMTX
        sample_rate: target sample rate after resampling (Changed default to 48kHz)
        channels: # of audio channels (Changed default to 2 for stereo Opus)
        buffer_seconds: max buffer duration stored
        """
        self.url = url
        self.sample_rate = sample_rate
        self.channels = channels

        # Correctly calculate buffer size for all channels
        self.buffer_samples = buffer_seconds * sample_rate * channels
        self.buffer = deque(maxlen=self.buffer_samples)

        self._running = False
        self._thread = None

    def start(self):
        """Start audio reading thread"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop audio reading thread"""
        self._running = False
        if self._thread:
            self._thread.join()

    def _worker(self):
        """
        Continuously pull audio from MediaMTX and store PCM samples into buffer
        """
        while self._running:
            try:
                # ### CHANGE 1: Added rtsp_transport: "tcp" to match your ffplay command ###
                # This is crucial for stability and avoiding packet loss over UDP.
                options = {"rtsp_transport": "tcp"}
                container = av.open(self.url, "r", options=options)
                
                # Find the audio stream. This works for Opus or any other codec.
                audio_stream = next(s for s in container.streams if s.type == "audio")

                resampler = av.audio.resampler.AudioResampler(
                    # ### FIX 1: Resample to float32 to avoid clipping ###
                    format="flt",
                    layout="mono" if self.channels == 1 else "stereo",
                    rate=self.sample_rate
                )

                for frame in container.decode(audio_stream):
                    if not self._running:
                        break

                    # PyAV decodes the Opus (or other) frame,
                    # and the resampler converts it to our target format (48kHz float32)
                    out_frames = resampler.resample(frame)

                    # Normalize to list
                    if not isinstance(out_frames, list):
                        out_frames = [out_frames]

                    for f in out_frames:
                        if f is None:
                            continue

                        # ### FIX 2: Store float32 samples directly ###
                        # .to_ndarray() will now return float32
                        pcm = f.to_ndarray().reshape(-1)
                        self.buffer.extend(pcm)

            except Exception as e:
                print("AudioTap: stream error:", e)
                if not self._running:
                    break
                print("Reconnecting in 1 second...")
                time.sleep(1)

    def get_last(self, seconds):
        """
        Retrieve the last N seconds of audio as float32 PCM (Whisper-ready)
        """
        # ### FIX 3: Get correct number of samples for all channels ###
        samples_needed = int(seconds * self.sample_rate * self.channels)
        if len(self.buffer) < samples_needed:
            return None

        # ### FIX 4: Just grab float32 data. No conversion needed. ###
        data = np.array(list(self.buffer)[-samples_needed:], dtype=np.float32)
        return data

    def get_segment(self, start_s, end_s):
        """
        Retrieve a specific slice.
        NOTE: Times are "seconds ago".
        get_segment(2, 5) -> gets audio from 5 seconds ago to 2 seconds ago.
        """
        if start_s < 0 or end_s < 0 or end_s <= start_s:
            print("AudioTap: invalid segment times.")
            return None

        # Corrected index math
        try:
            total_samples = len(self.buffer)
            start_idx_ago = int(end_s * self.sample_rate * self.channels)
            end_idx_ago = int(start_s * self.sample_rate * self.channels)

            if start_idx_ago > total_samples:
                print(f"AudioTap: Not enough data for segment. Need {start_idx_ago} samples, have {total_samples}")
                return None # Not enough data in buffer yet

            start_idx = total_samples - start_idx_ago
            end_idx = total_samples - end_idx_ago

            data = np.array(list(self.buffer)[start_idx:end_idx], dtype=np.float32)
            return data
        except Exception as e:
            print(f"AudioTap: Error getting segment: {e}")
            return None
            
    def save_mp3(self, segment_float32, path):
        segment_float32 = segment_float32.reshape(-1)

        pcm_int16 = np.clip(segment_float32 * 32768.0, -32768, 32767).astype(np.int16)

        container = av.open(path, mode="w")

        # MP3 stream – DO NOT set channels/layout
        stream = container.add_stream("mp3", rate=self.sample_rate)

        # Reshape for mono/stereo
        if self.channels == 1:
            pcm = pcm_int16.reshape(1, -1)
        else:
            pcm = pcm_int16.reshape(-1, self.channels).T

        frame_size = stream.codec_context.frame_size or 1152
        total_samples = pcm.shape[1]

        pos = 0
        while pos < total_samples:
            chunk = pcm[:, pos:pos + frame_size]
            pos += frame_size

            if chunk.shape[1] < frame_size:
                pad_width = frame_size - chunk.shape[1]
                chunk = np.pad(chunk, ((0,0),(0,pad_width)), mode='constant')

            frame = av.AudioFrame.from_ndarray(
                chunk,
                format="s16",
                layout="mono" if self.channels == 1 else "stereo"
            )

            frame.sample_rate = self.sample_rate

            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

        container.close()
