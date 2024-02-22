import sys
import time
import pyaudio
import numpy as np
from collections import deque

from utils.Numpy_Data_Buffer import Numpy_Data_Buffer


class Audio_Stream_Reader:
    def __init__(self,
                 device=None,
                 rate=None,
                 updates_per_second=1000,
                 verbose=False):

        self.rate = rate
        self.verbose = verbose
        self.pa = pyaudio.PyAudio()

        self.update_window_n_frames = 1024
        self.data_buffer = None

        self.device = device
        if self.device is None:
            self.device = self.input_device()
        if self.rate is None:
            self.rate = self.valid_low_rate(self.device)

        self.update_window_n_frames = round_up_to_even(
            self.rate / updates_per_second)
        self.updates_per_second = self.rate / self.update_window_n_frames
        self.info = self.pa.get_device_info_by_index(self.device)
        self.data_capture_delays = deque(maxlen=20)
        self.new_data = False
        if self.verbose:
            self.data_capture_delays = deque(maxlen=20)
            self.num_data_captures = 0

        self.stream = self.pa.open(
            input_device_index=self.device,
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.update_window_n_frames,
            stream_callback=self.non_blocking_stream_read)

    def non_blocking_stream_read(self, in_data, frame_count, time_info, status):
        if self.verbose:
            start = time.time()

        if self.data_buffer is not None:
            self.data_buffer.append_data(
                np.frombuffer(in_data, dtype=np.int16))
            self.new_data = True

        if self.verbose:
            self.num_data_captures += 1
            self.data_capture_delays.append(time.time() - start)

        return in_data, pyaudio.paContinue

    def stream_start(self, data_windows_to_buffer=None):
        self.data_windows_to_buffer = data_windows_to_buffer

        if data_windows_to_buffer is None:
            self.data_windows_to_buffer = int(self.updates_per_second / 2)
        else:
            self.data_windows_to_buffer = data_windows_to_buffer

        self.data_buffer = Numpy_Data_Buffer(
            self.data_windows_to_buffer, self.update_window_n_frames)

        print("\n-- Starting live audio stream...\n")
        self.stream.start_stream()
        self.stream_start_time = time.time()

    def terminate(self):
        print("Sending stream termination command...")
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def valid_low_rate(self, device, test_rates=[44100, 22050]):
        for testrate in test_rates:
            if self.test_device(device, rate=testrate):
                return testrate

        self.info = self.pa.get_device_info_by_index(device)
        default_rate = int(self.info["defaultSampleRate"])

        if self.test_device(device, rate=default_rate):
            return default_rate

        print(
            "SOMETHING'S WRONG! I can't figure out a good sample-rate for DEVICE =>", device)
        return default_rate

    def test_device(self, device, rate=None):
        try:
            self.info = self.pa.get_device_info_by_index(device)
            if not self.info["maxInputChannels"] > 0:
                return False

            if rate is None:
                rate = int(self.info["defaultSampleRate"])

            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                input_device_index=device,
                frames_per_buffer=self.update_window_n_frames,
                rate=rate,
                input=True)
            stream.close()
            return True
        except Exception as e:
            # print(e)
            return False

    def input_device(self):
        mics = []
        for device in range(self.pa.get_device_count()):
            if self.test_device(device):
                mics.append(device)

        if len(mics) == 0:
            print("No working microphone devices found!")
            sys.exit()

        print("Found %d working microphone device(s): " % len(mics))
        for mic in mics:
            self.print_mic_info(mic)

        return mics[0]

    def print_mic_info(self, mic):
        mic_info = self.pa.get_device_info_by_index(mic)
        print('\nMIC %s:' % (str(mic)))
        for k, v in sorted(mic_info.items()):
            print("%s: %s" % (k, v))
