import numpy as np
from scipy.signal import filtfilt, butter
from so_sorting.src import preprocessing



class AbstractProbe(object):
    """ An abstract probe encapsulates a large number of recording devices.
    Each of the instantiations of this abstract type can have a number of
    channels. Geometry must be specified for each subclass using the
    'get_neighbors' method specification.
    All AbstractProbes must implement several functions:
     sampling_rate() returns the sampling rate of the file
     num_channels() which returns the number of total channels (N)
     voltage_array which is an N x samples numpy array of raw recording.
     get_neighbors(index) # Returns a list of neighboring channels, which must
     include the input channel and be a numpy array of integers! """

    def __init__(self, sampling_rate, num_channels, voltage_array=None, voltage_dtype=None):
        self.sampling_rate = int(sampling_rate)
        self.num_channels = num_channels
        self.filter_band = [None, None]
        if voltage_dtype is None:
            voltage_dtype = voltage_array.dtype
        if voltage_array.dtype != voltage_dtype:
            print("Input voltage array is being cast from", voltage_array.dtype, "to", voltage_type, "because its type did not match the input voltage_type.")
            voltage_array = voltage_array.astype(voltage_type)
        self.v_dtype = voltage_dtype

        if voltage_array is not None:
            self.voltage = voltage_array
        else:
            # Set placeholder voltage with correct number of electrode channel
            # rows. Can be overwritten later with set_new_voltage()
            self.voltage = np.empty((num_channels, 0), dtype=self.v_dtype)

        # Voltage array must have one row for each channel with columns as samples
        if self.voltage.ndim == 1:
            # One dimensional voltage must be converted to 2 dimensions
            self.voltage = np.expand_dims(self.voltage, 0)
        if (self.num_channels != self.voltage.shape[0]) or (self.voltage.ndim < 1):
            raise ValueError("None of the voltage data dimensions match the input number of channels")
        self.n_samples = self.voltage.shape[1]

    def set_new_voltage(self, new_voltage_array):
        """ Reassigns voltage to new_voltage_array and updates dims/samples
        accordingly. """
        if not isinstance(new_voltage_array, np.ndarray):
            raise ValueError("Input new_voltage_array must be numpy ndarray")
        self.voltage = new_voltage_array
        # Voltage array must have one row for each channel with columns as samples
        if self.voltage.ndim == 1:
            self.voltage = np.expand_dims(self.voltage, 0)
        if (self.num_channels != self.voltage.shape[0]) or (self.voltage.ndim < 1):
            raise ValueError("None of the voltage data dimensions match the input number of channels")
        self.n_samples = self.voltage.shape[1]

    def get_voltage(self, channels, time_samples=None):
        """ Returns voltage trace for input channels and time sample indices.  These
            can be either tuples or slices. """
        # Channels are rows and time is columns
        if time_samples is None:
            time_samples = slice(0, self.voltage.shape[1], 1)
        return self.voltage[channels, time_samples]

    def set_voltage(self, channels, new_voltage, time_samples=None):
        """ Sets voltage trace for input channels and time sample indices.  These
            can be either tuples or slices. """
        # Channels are rows and time is columns
        if time_samples is None:
            time_samples = slice(0, self.voltage.shape[1], 1)
        self.voltage[channels, time_samples] = new_voltage

    def get_neighbors(channel):
        """ Should be defined by any subclass electrode/probe to account for
        their specific geometry. Must return numpy array of integers. """
        pass

    def bandpass_filter(self, low_cutoff=1000, high_cutoff=8000):
        """ One pole bandpass forward backward butterworth filter on each channel. """
        if self.filter_band[0] is not None and self.filter_band[1] is not None:
            if low_cutoff <= self.filter_band[0] and high_cutoff >= self.filter_band[1]:
                # Don't keep doing costly filtering if it won't have any effect
                print("Voltage has already been filtered within input frequency band. Skipping filtering.")
                return
        low_cutoff = low_cutoff / (self.sampling_rate / 2)
        high_cutoff = high_cutoff / (self.sampling_rate / 2)
        b_filt, a_filt = butter(1, [low_cutoff, high_cutoff], btype='band')
        for chan in range(0, self.num_channels):
            self.set_voltage(chan, filtfilt(b_filt, a_filt, self.get_voltage(chan), axis=0, padlen=None))

        # Update Probe filter band values
        if self.filter_band[0] is None:
            self.filter_band[0] = low_cutoff
        elif low_cutoff > self.filter_band[0]:
            self.filter_band[0] = low_cutoff
        if self.filter_band[1] is None:
            self.filter_band[1] = high_cutoff
        elif high_cutoff < self.filter_band[1]:
            self.filter_band[1] = high_cutoff


class SProbe16by2(AbstractProbe):

    def __init__(self, sampling_rate, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 32, voltage_array=voltage_array, voltage_dtype=None)

    def get_neighbors(self, channel):
        # These are organized into stereotrodes. Our neighbors are the channels on
        # our same stereotrode, the two stereotrodes above us, and the two
        # stereotrodes below us.

        if channel > self.num_channels - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        stereotrode_number = (channel) // 2
        total_stereotrodes = (32) // 2
        start_stereotrode = max(0, stereotrode_number - 1)
        end_stereotrode = min(total_stereotrodes, stereotrode_number + 2)
        neighbors = np.arange(start_stereotrode * 2, end_stereotrode * 2, 1)

        return np.int64(neighbors)


class SingleElectrode(AbstractProbe):

    def __init__(self, sampling_rate, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 1, voltage_array=voltage_array, voltage_dtype=None)

    def get_neighbors(self, channel):
        if channel > self.num_channels - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        return np.zeros(1, dtype=np.int64)


class SingleTetrode(AbstractProbe):

    def __init__(self, sampling_rate, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 4, voltage_array=voltage_array, voltage_dtype=None)

    def get_neighbors(self, channel):
        # These are organized into stereotrodes. Our neighbors are the channels on
        # our same stereotrode, the two stereotrodes above us, and the two
        # stereotrodes below us.

        if channel > self.num_channels - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        neighbors = np.arange(0, 4, 1)

        return np.int64(neighbors)


class DistanceBasedProbe(AbstractProbe):

    def __init__(self, sampling_rate, num_channels, xy_layout, voltage_array=None):
        """ xy_layout is 2D numpy array where each row represents its
        corresonding channel number and each column gives the x, y coordinates
        of that channel in micrometers. """
        AbstractProbe.__init__(self, sampling_rate, num_channels, voltage_array=voltage_array, voltage_dtype=None)

        self.distance_mat = np.zeros((xy_layout.shape[0], xy_layout.shape[0]))
        for n_trode in range(0, xy_layout.shape[0]):
            for n_pair in range(n_trode + 1, xy_layout.shape[0]):
                self.distance_mat[n_trode, n_pair] = np.sqrt(np.sum( \
                            (xy_layout[n_trode, :] - xy_layout[n_pair, :]) ** 2))
                self.distance_mat[n_pair, n_trode] = self.distance_mat[n_trode, n_pair]

    def get_neighbors(self, channel):
        """ Neighbors returned as all channels within 75 microns of input channel. """
        if channel > self.num_channels - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")
        neighbors = np.flatnonzero(self.distance_mat[channel, :] < 75)
        neighbors.sort()
        return np.int64(neighbors)


class SProbe24by1(AbstractProbe):

    def __init__(self, sampling_rate, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 24, voltage_array=voltage_array, voltage_dtype=None)

    def get_neighbors(self, channel):
        if channel > self.num_channels - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")
        start_channel = max(0, channel - 1)
        stop_channel = min(24, channel + 2)
        neighbors = np.arange(start_channel, stop_channel)
        return np.int64(neighbors)
