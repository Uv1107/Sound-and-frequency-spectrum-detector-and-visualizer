import math
import time
import numpy as np
from collections import deque
from scipy.signal import savgol_filter

from utils.helpers import round_up_to_even
from utils.Numpy_Data_Buffer import Numpy_Data_Buffer


class Audio_Stream_Analyzer:
    def __init__(self,
                 device=None,
                 rate=None,
                 audio_window_size_ms=50,
                 updates_per_second=100,
                 smoothing_length_ms=50,
                 n_frequency_bins=51,
                 visualize=True,
                 verbose=False,
                 height=450,
                 window_ratio=24/9):

        self.n_frequency_bins = n_frequency_bins
        self.rate = rate
        self.verbose = verbose
        self.visualize = visualize
        self.height = height
        self.window_ratio = window_ratio

        try:
            from audio_stream_reader_pyaudio import Audio_Stream_Reader
            self.stream_reader = Audio_Stream_Reader(
                device=device,
                rate=rate,
                updates_per_second=updates_per_second,
                verbose=verbose)
        except:
            from src.audio_stream_reader_sounddevice import Audio_Stream_Reader
            self.stream_reader = Audio_Stream_Reader(
                device=device,
                rate=rate,
                verbose=verbose)

        self.rate = self.stream_reader.rate

        self.rolling_stats_window_s = 20
        self.equalizer_strength = 0.20
        self.apply_frequency_smoothing = True

        if self.apply_frequency_smoothing:
            self.filter_width = round_up_to_even(
                0.03*self.n_frequency_bins) - 1
        if self.visualize:
            from src.audio_visualizer import Audio_Spectrum_Visualizer

        self.audio_window_size = round_up_to_even(
            self.rate * audio_window_size_ms / 1000)
        self.audio_window_size_ms = 1000 * self.audio_window_size / self.rate
        self.audio = np.ones(int(self.audio_window_size/2), dtype=float)
        self.audiox = np.arange(
            int(self.audio_window_size/2), dtype=float) * self.rate / self.audio_window_size

        self.data_windows_to_buffer = math.ceil(
            self.audio_window_size / self.stream_reader.update_window_n_frames)
        self.data_windows_to_buffer = max(1, self.data_windows_to_buffer)

        self.smoothing_length_ms = smoothing_length_ms
        if self.smoothing_length_ms > 0:
            self.smoothing_kernel = self.get_smoothing_filter(
                self.audio_window_size_ms, self.smoothing_length_ms)
            self.feature_buffer = Numpy_Data_Buffer(len(self.smoothing_kernel), len(
                self.audio), dtype=np.float32, data_dimensions=2)

        self.audiox_bin_indices = np.logspace(np.log2(len(self.audiox)), 0, len(
            self.audiox), endpoint=True, base=2, dtype=None) - 1
        self.audiox_bin_indices = np.round(((self.audiox_bin_indices - np.max(
            self.audiox_bin_indices))*-1) / (len(self.audiox) / self.n_frequency_bins), 0).astype(int)
        self.audiox_bin_indices = np.minimum(np.arange(len(
            self.audiox_bin_indices)), self.audiox_bin_indices - np.min(self.audiox_bin_indices))

        self.frequency_bin_energies = np.zeros(self.n_frequency_bins)
        self.frequency_bin_centres = np.zeros(self.n_frequency_bins)
        self.audiox_indices_per_bin = []
        for bin_index in range(self.n_frequency_bins):
            bin_frequency_indices = np.where(
                self.audiox_bin_indices == bin_index)
            self.audiox_indices_per_bin.append(bin_frequency_indices)
            audiox_frequencies_this_bin = self.audiox[bin_frequency_indices]
            self.frequency_bin_centres[bin_index] = np.mean(
                audiox_frequencies_this_bin)

        self.audio_fps = 30
        self.log_features = False
        self.delays = deque(maxlen=20)
        self.num_audios = 0
        self.strongest_frequency = 0

        self.power_normalization_coefficients = np.logspace(np.log2(1), np.log2(
            np.log2(self.rate/2)), len(self.audiox), endpoint=True, base=2, dtype=None)
        self.rolling_stats_window_n = self.rolling_stats_window_s * \
            self.audio_fps
        self.rolling_bin_values = Numpy_Data_Buffer(
            self.rolling_stats_window_n, self.n_frequency_bins, start_value=25000)
        self.bin_mean_values = np.ones(self.n_frequency_bins)

        self.stream_reader.stream_start(self.data_windows_to_buffer)

        if self.visualize:
            self.visualizer = Audio_Spectrum_Visualizer(self)
            self.visualizer.start()

    def get_audio(self, data, rate, chunk_size, log_scale=False):
        data = data * np.hamming(len(data))
        try:
            audio = np.abs(np.fft.rfft(data)[1:])
        except:
            audio = np.fft.fft(data)
            left, right = np.split(np.abs(audio), 2)
            audio = np.add(left, right[::-1])

        if log_scale:
            try:
                audio = np.multiply(20, np.log10(audio))
            except Exception as e:
                print('Log(Audio) failed: %s' % str(e))

        return audio

    def gaussian_kernel_1D(self, sigma, truncate=2.0):
        sigma = float(sigma)
        sigma2 = sigma * sigma
        radius = int(truncate * sigma + 0.5)

        x = np.arange(-radius, radius+1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()
        return phi_x

    def get_smoothing_filter(self, audio_window_size_ms, filter_length_ms):
        buffer_length = round_up_to_even(
            filter_length_ms / audio_window_size_ms)+1
        filter_sigma = buffer_length / 3
        filter_weights = self.gaussian_kernel_1D(filter_sigma)[:, np.newaxis]

        max_index = np.argmax(filter_weights)
        filter_weights = filter_weights[:max_index+1]
        filter_weights = filter_weights / np.mean(filter_weights)

        return filter_weights

    def update_rolling_stats(self):
        self.rolling_bin_values.append_data(self.frequency_bin_energies)
        self.bin_mean_values = np.mean(
            self.rolling_bin_values.get_buffer_data(), axis=0)
        self.bin_mean_values = np.maximum(
            (1-self.equalizer_strength)*np.mean(self.bin_mean_values), self.bin_mean_values)

    def update_features(self):

        latest_data_window = self.stream_reader.data_buffer.get_most_recent(
            self.audio_window_size)

        self.audio = self.get_audio(latest_data_window, self.rate,
                                    self.audio_window_size, log_scale=self.log_features)

        self.audio = self.audio * self.power_normalization_coefficients
        self.num_audios += 1
        self.audio_fps = self.num_audios / \
            (time.time() - self.stream_reader.stream_start_time)

        if self.smoothing_length_ms > 0:
            self.feature_buffer.append_data(self.audio)
            buffered_features = self.feature_buffer.get_most_recent(
                len(self.smoothing_kernel))
            if len(buffered_features) == len(self.smoothing_kernel):
                buffered_features = self.smoothing_kernel * buffered_features
                self.audio = np.mean(buffered_features, axis=0)

        self.strongest_frequency = self.audiox[np.argmax(self.audio)]

        for bin_index in range(self.n_frequency_bins):
            self.frequency_bin_energies[bin_index] = np.mean(
                self.audio[self.audiox_indices_per_bin[bin_index]])

        return

    def get_audio_features(self):

        if self.stream_reader.new_data:
            if self.verbose:
                start = time.time()

            self.update_features()
            self.update_rolling_stats()
            self.stream_reader.new_data = False

            self.frequency_bin_energies = np.nan_to_num(
                self.frequency_bin_energies, copy=True)
            if self.apply_frequency_smoothing:
                if self.filter_width > 3:
                    self.frequency_bin_energies = savgol_filter(
                        self.frequency_bin_energies, self.filter_width, 3)
            self.frequency_bin_energies[self.frequency_bin_energies < 0] = 0

            if self.verbose:
                self.delays.append(time.time() - start)

            if self.visualize and self.visualizer._is_running:
                self.visualizer.update()

        return
