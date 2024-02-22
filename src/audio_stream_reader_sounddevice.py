import time
import numpy as np
import sounddevice as sd
from collections import deque

from utils.helpers import round_up_to_even
from utils.Numpy_Data_Buffer import Numpy_Data_Buffer


class Audio_Stream_Reader:
    def __init__(self,
                 device=None,
                 rate=None,
                 verbose=False):

        device_dict = sd.query_devices()

        try:
            sd.check_input_settings(
                device=device, channels=1, dtype=np.float32, extra_settings=None, samplerate=rate)
        except:
            rate = None
            device = None

        self.rate = rate
        if rate is not None:
            sd.default.samplerate = rate

        self.device = device
        if device is not None:
            sd.default.device = device

        self.verbose = verbose
        self.data_buffer = None

        self.optimal_data_lengths = []
        with sd.InputStream(samplerate=self.rate,
                            blocksize=0,
                            device=self.device,
                            channels=1,
                            dtype=np.float32,
                            latency='low',
                            callback=self.test_stream_read):
            time.sleep(0.2)

        self.update_window_n_frames = max(self.optimal_data_lengths)
        del self.optimal_data_lengths

        self.stream = sd.InputStream(
            samplerate=self.rate,
            blocksize=self.update_window_n_frames,
            device=None,
            channels=1,
            dtype=np.float32,
            latency='low',
            extra_settings=None,
            callback=self.non_blocking_stream_read)

        self.rate = self.stream.samplerate
        self.device = self.stream.device

        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = ''
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.device_latency = device_dict[self.device]['default_low_input_latency']

    def non_blocking_stream_read(self, indata, frames, time_info, status):
        if self.verbose:
            start = time.time()
            if status:
                print(status)

        if self.data_buffer is not None:
            self.data_buffer.append_data(indata[:, 0])
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return

    def test_stream_read(self, indata, frames, time_info, status):
        self.optimal_data_lengths.append(len(indata[:, 0]))
        return

    def stream_start(self, data_windows_to_buffer=None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2)
        else:
            self.data_windows_to_buffer = data_windows_to_buffer

        self.data_buffer = Numpy_Data_Buffer(
            self.data_windows_to_buffer, self.update_window_n_frames)

        self.stream.start()
        self.stream_start_time = time.time()

    def terminate(self):
        self.stream.stop()
