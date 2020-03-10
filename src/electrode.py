import numpy as np
from scipy import signal
from spikesorting_python.src import preprocessing


""" Translation of David's Julia code into Python. """


class AbstractProbe(object):
    """ An abstract probe encapsulates a large number of recording devices.
    Each of the instantiations of this abstract type can have a number of
    electrodes and/or a geometry.
    All AbstractProbes must implement several functions:
     sampling_rate() returns the sampling rate of the file
     num_electrodes() which returns the number of total electrodes (N)
     voltage(, [channel]) which returns the voltage trace for a given electrode
     get_neighbors(index) # Returns a list of neighboring channels, which must
     include the input channel! """

    def __init__(self, sampling_rate, num_electrodes, fname_voltage=None, voltage_array=None):
        self.sampling_rate = int(sampling_rate)
        self.num_electrodes = num_electrodes
        # self.reader = SpikeReader
        self.fname_voltage = None
        self.filter_band = [None, None]

        if voltage_array is not None and fname_voltage is None:
            # If voltage array is given use it even if a filename is input
            if not isinstance(voltage_array, np.ndarray):
                raise ValueError("Input voltage_array must be numpy ndarray")
            self.voltage = voltage_array
        elif fname_voltage is not None and voltage_array is None:
            # Only a filename is given so memory map it
            if fname_voltage[-4:] != ".npy":
                fname_voltage = fname_voltage + ".npy"
            self.voltage = np.load(fname_voltage, mmap_mode='r+', allow_pickle=False, fix_imports=False)
            self.fname_voltage = fname_voltage
        elif voltage_array is not None and fname_voltage is not None:
            raise ValueError("Input voltage must be only one of fname_voltage OR voltage_array, but not both.")
        else:
            # Set placeholder voltage with correct number of electrode rows and dims
            self.voltage = np.empty((num_electrodes, 0))

        # Voltage array must have one row for each channel with columns as samples
        if self.voltage.ndim != 2:
            self.voltage = np.expand_dims(self.voltage, 0)
        if self.num_electrodes != self.voltage.shape[0]:
            self.voltage = self.voltage.T
        if (self.num_electrodes != self.voltage.shape[0]) or (self.voltage.ndim < 1):
            raise ValueError("None of the voltage data dimensions match the input number of electrodes")
        self.n_samples = self.voltage.shape[1]

    def get_voltage(self, channels, time_samples=None):
        """ Returns voltage trace for input channels and time sample indices.  These
            can be either tuples or slices. """
        # Electrodes are rows and time is columns
        if time_samples is None:
            time_samples = slice(0, self.voltage.shape[1], 1)
        return self.voltage[channels, time_samples]

    def set_voltage(self, channels, new_voltage, time_samples=None):
        """ Sets voltage trace for input channels and time sample indices.  These
            can be either tuples or slices. """
        # Electrodes are rows and time is columns
        if time_samples is None:
            time_samples = slice(0, self.voltage.shape[1], 1)
        self.voltage[channels, time_samples] = new_voltage

    def get_neighbors(channel):
        pass

    def bandpass_filter(self, low_cutoff=300, high_cutoff=6000):

        if self.filter_band[0] is not None and self.filter_band[1] is not None:
            if low_cutoff <= self.filter_band[0] and high_cutoff >= self.filter_band[1]:
                # Don't keep doing costly filtering if it won't have any effect
                return
        low_cutoff = low_cutoff / (self.sampling_rate / 2)
        high_cutoff = high_cutoff / (self.sampling_rate / 2)
        b_filt, a_filt = signal.butter(1, [low_cutoff, high_cutoff], btype='band')
        for chan in range(0, self.num_electrodes):
            self.set_voltage(chan, signal.filtfilt(b_filt, a_filt, self.get_voltage(chan), axis=0, padlen=None))

        if self.filter_band[0] is None:
            self.filter_band[0] = -np.inf
        if low_cutoff > self.filter_band[0]:
            self.filter_band[0] = low_cutoff
        if self.filter_band[1] is None:
            self.filter_band[1] = np.inf
        if high_cutoff < self.filter_band[1]:
            self.filter_band[1] = high_cutoff

    # def zca_transform(self):
    #     zca_matrix = preprocessing.get_noise_sampled_zca_matrix(self.voltage)
    #     zca_data = np.empty((self.num_electrodes, self.n_samples), dtype='float64', order='C')
    #     zca_data = np.matmul(zca_matrix, self.voltage, out=zca_data)
    #     self.voltage = zca_data


class SProbe16by2(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 32, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        # These are organized into stereotrodes. Our neighbors are the channels on
        # our same stereotrode, the two stereotrodes above us, and the two
        # stereotrodes below us.

        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        stereotrode_number = (channel) // 2
        total_stereotrodes = (32) // 2
        start_stereotrode = max(0, stereotrode_number - 1)
        end_stereotrode = min(total_stereotrodes, stereotrode_number + 2)
        neighbors = np.arange(start_stereotrode * 2, end_stereotrode * 2, 1)

        return neighbors


class SingleElectrode(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 1, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        return np.zeros(1, dtype=np.int64)


class SingleTetrode(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 4, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        # These are organized into stereotrodes. Our neighbors are the channels on
        # our same stereotrode, the two stereotrodes above us, and the two
        # stereotrodes below us.

        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        neighbors = np.arange(0, 4, 1)

        return neighbors


class Dense32Probe(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 32, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):

        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        middle_channel_numbers = [22, 9, 23, 8, 17, 14, 16, 15, 30, 1, 31, 0]
        coordinates = []
        for row in range(0, len(middle_channel_numbers)):
            coordinates.append([0, 25*row])
        left_channel_numbers = [29, 28, 18, 27, 19, 26, 20, 25, 21, 24]
        for row in range(0, len(left_channel_numbers)):
            coordinates.append([-18, 25*row + 12.5])
        right_channel_numbers = [2, 3, 13, 4, 12, 5, 11, 6, 10, 7]
        for row in range(0, len(right_channel_numbers)):
            coordinates.append([18, 25*row + 12.5])
        coordinates = np.array(coordinates)
        all_channel_numbers = middle_channel_numbers
        all_channel_numbers.extend(left_channel_numbers)
        all_channel_numbers.extend(right_channel_numbers)
        all_channel_numbers = np.array(all_channel_numbers)

        distance_mat = np.empty((coordinates.shape[0], coordinates.shape[0]))
        for xy in range(coordinates.shape[0]):
            distance_mat[xy, :] = np.sqrt(np.sum((coordinates - coordinates[xy, :]) ** 2, axis=1))

        neighbors = all_channel_numbers[distance_mat[np.argwhere(all_channel_numbers == channel)[0][0], :] <= 50]

        return neighbors


class Dense128Probe(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 128, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):

        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        all_channel_numbers = [ 20,  16,  23,  17,  22,  19,  25,  18,  24,  21,  27,  15,  26,
        14,  29,  13,  28,  12,  31,  11,  30,  10,  33,   9,  32,   8,
        35,   7,  34,   6,  37,   5,  36,   4,  39,   3,  57,  58,  59,
        60,  61,  62,  63,   0,   1,   2,  38,  56,  41,  55,  40,  54,
        43,  53,  42,  52,  45,  51,  44,  50,  47,  49,  46,  48, 110,
        112, 111, 113, 108, 114,  81,  79,  80,  78,  83,  77,  82,  76,
        85,  75,  84,  74,  65,  64,  66,  67,  68,  69,  70,  71,  72,
        73,  87, 127,  86, 126,  89, 125,  88, 124,  91, 123,  90, 122,
        93, 121,  92, 120,  95, 119,  94, 118,  97, 117,  96, 116,  99,
        115,  98, 105, 101, 104, 100, 107, 103, 106, 102, 109]
        coordinates = []
        for y in range(0, 32):
            for x in range(0, 4):
                coordinates.append([x * 22.5, -y * 22.5])
        coordinates = np.array(coordinates)
        all_channel_numbers = np.array(all_channel_numbers)

        distance_mat = np.empty((coordinates.shape[0], coordinates.shape[0]))
        for xy in range(coordinates.shape[0]):
            distance_mat[xy, :] = np.sqrt(np.sum((coordinates - coordinates[xy, :]) ** 2, axis=1))

        neighbors = all_channel_numbers[distance_mat[np.argwhere(all_channel_numbers == channel)[0][0], :] <= 40]

        return neighbors.sort()


class SProbe24by1(AbstractProbe):

    def __init__(self, sampling_rate, fname_voltage=None, voltage_array=None):
        AbstractProbe.__init__(self, sampling_rate, 24, fname_voltage=fname_voltage, voltage_array=voltage_array)

    def get_neighbors(self, channel):
        # These are organized into stereotrodes. Our neighbors are the channels on
        # our same stereotrode, the two stereotrodes above us, and the two
        # stereotrodes below us.

        if channel > self.num_electrodes - 1 or channel < 0:
            raise ValueError("Invalid electrode channel")

        start_electrode = max(0, channel - 1)
        stop_electrode = min(24, channel + 2)
        neighbors = np.arange(start_electrode, stop_electrode)

        return neighbors


def voltage_from_PL2(PL2Reader):
    include_chans = []
    for cc in range(0, len(PL2Reader.info['analog_channels'])):
        if PL2Reader.info['analog_channels'][cc]['source_name'].upper() == 'SPKC' and bool(PL2Reader.info['analog_channels'][cc]['enabled']):
            include_chans.append(cc)

    voltage = np.empty((len(include_chans), PL2Reader.info['analog_channels'][include_chans[0]]['num_values']), dtype='int16', order='C')
    v_index = 0
    for chan in include_chans:
        PL2Reader.load_analog_channel(chan, convert_to_mv=False, out=voltage[v_index, :])
        v_index += 1

    return voltage


def voltage_to_npy(PL2Reader, save_fname=None):
    """ This reads in all enabled analog channels with 'source_name' = 'SPKC' and saves
        them in a single numpy array of size n_samples x channels in column major order.
        Data are saved in the order they are found, which should be their neighboring
        locations. Default saves in same place as PL2 file. """

    if save_fname is None:
        save_fname = PL2Reader.filename[0:-4]

    include_chans = []
    for cc in range(0, len(PL2Reader.info['analog_channels'])):
        if PL2Reader.info['analog_channels'][cc]['source_name'].upper() == 'SPKC' and bool(PL2Reader.info['analog_channels'][cc]['enabled']):
            include_chans.append(cc)

    voltage = np.empty((PL2Reader.info['analog_channels'][include_chans[0]]['num_values'], len(include_chans)), dtype='int16', order='C')
    v_index = 0
    for chan in include_chans:
        PL2Reader.load_analog_channel(chan, convert_to_mv=False, out=voltage[:, v_index])
        v_index += 1

    np.save(save_fname, voltage, allow_pickle=False, fix_imports=False)


def compute_full_contiguous_zca(PL2Reader, low_cutoff=200, high_cutoff=8000):
    """
    """

    include_chans = []
    for cc in range(0, len(PL2Reader.info['analog_channels'])):
        if PL2Reader.info['analog_channels'][cc]['source_name'].upper() == 'SPKC' and bool(PL2Reader.info['analog_channels'][cc]['enabled']):
            include_chans.append(cc)

    # Read data
    voltage = np.empty((len(include_chans), PL2Reader.info['analog_channels'][include_chans[0]]['num_values']), dtype='int16', order='C')
    v_index = 0
    for chan in include_chans:
        PL2Reader.load_analog_channel(chan, convert_to_mv=False, out=voltage[v_index, :])
        v_index += 1

    # Filter data, if filter ranges are not beyond Plexon hardware
    if low_cutoff > 250 or high_cutoff < 8000:
        low_cutoff = low_cutoff / (PL2Reader.info['timestamp_frequency'] / 2)
        high_cutoff = high_cutoff / (PL2Reader.info['timestamp_frequency'] / 2)
        b_filt, a_filt = signal.butter(1, [low_cutoff, high_cutoff], btype='band')
        for chan in range(0, voltage.shape[0]):
            voltage[chan, :] = signal.filtfilt(b_filt, a_filt, voltage[chan, :], padlen=None)

    zca_matrix = preprocessing.get_zca_matrix(voltage, rowvar=True)
    zca_data = np.empty((voltage.shape[0], voltage.shape[1]), dtype='float64', order='C')
    zca_data = np.matmul(zca_matrix, voltage, out=zca_data)

    return zca_data
