import numpy as np
from scipy import signal
from scipy import stats
from spikesorting_fullpursuit import neuron_separability
from spikesorting_fullpursuit.parallel import binary_pursuit_parallel
from spikesorting_fullpursuit.analyze_spike_timing import find_overlapping_spike_bool
from spikesorting_fullpursuit.parallel.segment_parallel import identify_threshold_crossings, time_window_to_samples, median_threshold
from spikesorting_fullpursuit.utils.parallel_funs import noise_covariance_parallel

from scipy.stats import norm
import matplotlib.pyplot as plt
import spikesorting_fullpursuit.parallel.segment_parallel



dict_other_templates = {'upsidedown_47a': np.array(
      [-0.1513909 , -0.16631116, -0.17847979, -0.18862691, -0.19872002,
       -0.20880426, -0.21737931, -0.22361187, -0.22874592, -0.23452372,
       -0.24082686, -0.24489726, -0.24255492, -0.22836725, -0.19307666,
       -0.1191242 ,  0.02425   ,  0.28417833,  0.71488001,  1.34970811,
        2.15717331,  3.00861469,  3.70470693,  4.06647289,  4.02591484,
        3.64727615,  3.07093063,  2.43142789,  1.8080954 ,  1.2247536 ,
        0.67450634,  0.14624459, -0.36222091, -0.84091953, -1.2741004 ,
       -1.6452294 , -1.94012256, -2.14839704, -2.26756383, -2.30440089,
       -2.27405578, -2.19505695, -2.0840863 , -1.95470787, -1.81895549,
       -1.68644032, -1.56235016, -1.44811373, -1.3434583 , -1.24678163,
       -1.15489337, -1.06493637, -0.97600263, -0.88764435, -0.79968051,
       -0.71323625, -0.62895673, -0.54701365, -0.46780424, -0.39222975,
       -0.32111773, -0.25396762, -0.19020192, -0.13038235, -0.07453424,
       -0.02132602,  0.02993237,  0.07660115,  0.11522204,  0.14610511,
        0.17122906,  0.19176712,  0.20754389]),
        'upsidedown_47b': np.array(
        [-0.1513909 , -0.16631116, -0.17847979, -0.18862691, -0.19872002,
       -0.20880426, -0.21737931, -0.22361187, -0.22874592, -0.23452372,
       -0.24082686, -0.24489726, -0.24255492, -0.22836725, -0.19307666,
       -0.1191242 ,  0.02425   ,  0.28417833,  0.71488001,  1.34970811,
        2.15717331,  3.00861469,  3.70470693,  4.06647289,  4.02591484,
        3.64727615,  3.07093063,  2.43142789,  1.8080954 ,  1.2247536 ,
        0.67450634,  0.14624459, -0.36222091, -0.84091953, -1.2741004 ,
       -1.6452294 , -1.94012256, -2.14839704, -2.26756383, -2.30440089,
       -2.27405578, -2.19505695, -2.0840863 , -1.95470787, -1.81895549,
       -1.68644032, -1.56235016, -1.44811373, -1.3434583 , -1.24678163,
       -1.15489337, -1.06493637, -0.97600263, -0.88764435, -0.79968051,
       -0.71323625, -0.62895673, -0.54701365, -0.46780424, -0.39222975,
       -0.32111773, -0.25396762, -0.19020192, -0.13038235, -0.07453424,
       -0.02132602,  0.02993237,  0.07660115,  0.11522204,  0.14610511,
        0.17122906,  0.19176712,  0.20754389])}


class RunBinaryPursuit(object):
    """
    Any input templates will be subsequently multiplied by 'amplitude' to keep
    them in scale with the noise.
    Actual IDs are output as the index at (self.neuron_templates.shape[1] // 2)
    """
    def __init__(self, num_channels, duration,
                 neuron_templates=None, frequency_range=(500, 8000),
                 samples_per_second=40000, amplitude=1, percent_shared_noise=0,
                 correlate1_2=False, voltage_dtype=np.float32):
        self.num_channels = num_channels
        self.duration = duration
        self.voltage_array = None
        self.frequency_range = frequency_range
        self.samples_per_second = samples_per_second
        self.voltage_dtype = voltage_dtype
        self.correlate1_2 = correlate1_2

        self.amplitude = amplitude
        if neuron_templates is not None:
            self.neuron_templates = neuron_templates
        else:
            self.neuron_templates = self.get_default_templates()
        self.neuron_templates = (self.neuron_templates * self.amplitude).astype(self.voltage_dtype)
        self.actual_IDs = []
        self.percent_shared_noise = percent_shared_noise
        if self.percent_shared_noise > 1:
            self.percent_shared_noise = 1.
        if self.percent_shared_noise < 0.:
            self.percent_shared_noise = 0.

    def get_default_templates(self):
        templates = np.array([
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.001, -0.0, -0.0, 0.0, 0.0, 0.001, 0.002, 0.005, 0.01, 0.019, 0.034, 0.057, 0.1, 0.166, 0.222, 0.167, -0.088, -0.49, -0.855, -1.0, -0.862, -0.526, -0.151, 0.136, 0.293, 0.339, 0.312, 0.251, 0.187, 0.138, 0.109, 0.093, 0.084, 0.076, 0.069, 0.063, 0.058, 0.056, 0.054, 0.052, 0.051, 0.05, 0.049, 0.049, 0.048, 0.048, 0.047, 0.047, 0.046, 0.046, 0.045, 0.044, 0.043, 0.042, 0.041, 0.04, 0.039, 0.038, 0.037, 0.036, 0.035, 0.034, 0.033, 0.032, 0.03, 0.029, 0.027, 0.026, 0.025],
            [0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.001, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005, 0.005, 0.006, 0.007, 0.007, 0.008, 0.009, 0.011, 0.013, 0.017, 0.024, 0.033, 0.045, 0.062, 0.084, 0.091, 0.033, -0.166, -0.513, -0.858, -1.0, -0.868, -0.555, -0.212, 0.057, 0.218, 0.285, 0.293, 0.274, 0.248, 0.224, 0.204, 0.187, 0.173, 0.16, 0.149, 0.14, 0.132, 0.124, 0.117, 0.11, 0.103, 0.097, 0.091, 0.086, 0.081, 0.076, 0.072, 0.067, 0.064, 0.06, 0.056, 0.053, 0.05, 0.046, 0.043, 0.04, 0.038, 0.035, 0.033, 0.031, 0.028, 0.026, 0.023, 0.021, 0.019, 0.017, 0.015, 0.013, 0.012],
            [-0.108, -0.11, -0.111, -0.112, -0.112, -0.11, -0.107, -0.105, -0.104, -0.101, -0.099, -0.097, -0.095, -0.092, -0.088, -0.083, -0.077, -0.071, -0.065, -0.062, -0.058, -0.053, -0.045, -0.037, -0.03, -0.022, -0.013, -0.003, 0.01, 0.027, 0.045, 0.059, 0.071, 0.081, 0.092, 0.106, 0.12, 0.133, 0.145, 0.158, 0.171, 0.19, 0.215, 0.245, 0.276, 0.305, 0.351, 0.453, 0.642, 0.877, 1.0, 0.883, 0.647, 0.45, 0.342, 0.295, 0.269, 0.242, 0.214, 0.189, 0.171, 0.158, 0.146, 0.134, 0.12, 0.107, 0.094, 0.083, 0.072, 0.06, 0.049, 0.036, 0.022, 0.008, -0.006, -0.019, -0.027, -0.033, -0.04, -0.048, -0.054, -0.059, -0.063, -0.069, -0.073, -0.079, -0.086, -0.092, -0.095, -0.096, -0.098, -0.101, -0.105, -0.105, -0.105, -0.105, -0.105, -0.107, -0.109, -0.111]
            ])
        # Rescale templates from -1 to 1
        templates /= np.amax(np.abs(templates), axis=1)[:, None]
        return templates

    def gen_poisson_spiketrain(self, firing_rate, tau_ref=1.5e-3):
        """gen_poission_spiketrain(rate, refactory period, samples per second [, duration])
        Generate a poisson distributed neuron spike train given the passed firing rate (in Hz).
        The refactory period (in sec) determines the silency period between two adjacent spikes.
        Samples per seconds defines the sampling rate of output spiketrain. """
        number_of_samples = int(np.round(self.duration * self.samples_per_second))
        spiketrain = np.zeros(number_of_samples, dtype=np.bool)
        random_nums = np.random.random(spiketrain.size)

        if tau_ref < 0:
            tau_ref = 0
        elif tau_ref > 0:
            # Adjust the firing rate to take into account the refactory period
            firing_rate = firing_rate * 1.0 / (1.0 - firing_rate * tau_ref)

        # Generate the spiketrain
        last_spike_index = -np.inf
        # Do not add any spikes for the first and last 5 ms
        for i in range(int(5*self.samples_per_second / 1000), spiketrain.size - int(5*self.samples_per_second / 1000)):
            if (i - last_spike_index) / self.samples_per_second > tau_ref:
                spiketrain[i] = random_nums[i] < firing_rate / self.samples_per_second
                if spiketrain[i]:
                    last_spike_index = i
        return spiketrain

    def gen_bandlimited_noise(self):
        """gen_bandlimited_noise([frequency_range, samples_per_second, duration, amplitude])
        Generates a timeseries of bandlimited Gaussian noise. This function uses the inverse FFT to generate
        noise in a given bandwidth. Within this bandwidth, the amplitude of each frequency is approximately
        constant (i.e., white), but the phase is random. The amplitude of the output signal is scaled so that
        99% of the values fall within the amplitude criteron (3 standard deviations of zero mean)."""
        freqs = np.abs(np.fft.fftfreq(self.samples_per_second * self.duration, 1/self.samples_per_second))
        f = np.zeros(self.samples_per_second * self.duration)
        idx = np.where(np.logical_and(freqs>=self.frequency_range[0], freqs<=self.frequency_range[1]))[0]
        f[idx] = 1
        f = np.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        bandlimited_noise = np.fft.ifft(f).real
        bandlimited_noise = bandlimited_noise * self.amplitude / (3 * np.std(bandlimited_noise))
        bandlimited_noise = bandlimited_noise.astype(self.voltage_dtype)
        return bandlimited_noise

    def gen_noise_voltage_array(self):
        self.voltage_array = np.zeros((self.num_channels, int(self.samples_per_second*self.duration)), dtype=self.voltage_dtype)
        if self.percent_shared_noise > 0:
            # Make a shared array. There are other ways this could be done, but here
            # all channels get percent_shared_noise amplitude shared noise plus
            # 1 - percent_shared_noise independent noise
            shared_noise = self.percent_shared_noise * self.gen_bandlimited_noise()
            for i in range (0, self.num_channels):
                self.voltage_array[i, :] = shared_noise
        for i in range (0, self.num_channels):
            self.voltage_array[i, :] += ((1-self.percent_shared_noise) * self.gen_bandlimited_noise()).astype(self.voltage_dtype)

    def gen_test_dataset(self, firing_rates, template_inds, chan_scaling_factors,
                         refractory_wins=1.5e-3, remove_overlaps=False):
        """ Creates the test data set by making a noise voltage array and adding
        in spikes for the neurons in template_inds with corresponding rates and
        scaling.
        """
        if len(firing_rates) != len(template_inds) or len(firing_rates) != len(chan_scaling_factors):
            raise ValueError("Input rates, templates, and scaling factors must all be the same length!")

        # Need to clear old voltage if spikes have been assigned
        if len(self.actual_IDs) > 0:
            self.voltage_array = None
        low_cutoff = self.frequency_range[0] / (self.samples_per_second / 2)
        high_cutoff = self.frequency_range[1] / (self.samples_per_second / 2)
        b_filt, a_filt = signal.butter(1, [low_cutoff, high_cutoff], btype='band')
        refractory_wins = np.array(refractory_wins)
        if refractory_wins.size == 1:
            refractory_wins = np.repeat(refractory_wins, len(firing_rates))
        # Reset neuron actual IDs for each neuron
        self.actual_IDs = [[] for neur in range(0, len(firing_rates))]
        self.actual_templates = [np.zeros((1, 0), dtype=self.voltage_dtype) for neur in range(0, len(firing_rates))]
        if self.voltage_array is None:
            self.gen_noise_voltage_array()
        for neuron in range(0, len(firing_rates)):
            # Generate one set of spike times for each neuron
            spiketrain = self.gen_poisson_spiketrain(firing_rate=firing_rates[neuron], tau_ref=refractory_wins[neuron])
            self.actual_IDs[neuron] = np.where(spiketrain)[0]

        if remove_overlaps:
            # Remove any spike times that might overlap with each other within half a template
            overlapping_bools = []
            for n1 in range(0, len(self.actual_IDs)):
                overlapping_bools.append(np.ones(self.actual_IDs[n1].shape[0], dtype=np.bool))
                for n2 in range(0, len(self.actual_IDs)):
                    if n1 == n2:
                        continue
                    overlapping_spike_bool = find_overlapping_spike_bool(
                            self.actual_IDs[n1], self.actual_IDs[n2], overlap_tol=self.neuron_templates[0].shape[0])
                    overlapping_bools[n1] = np.logical_and(overlapping_bools[n1], ~overlapping_spike_bool)
            for ob in range(0, len(self.actual_IDs)):
                self.actual_IDs[ob] = self.actual_IDs[ob][overlapping_bools[ob]]


        for neuron in range(0, len(firing_rates)):
            # Set boolean spike train from spike times. spiketrain was made above
            # so just zero it out here
            spiketrain[:] = False
            spiketrain[self.actual_IDs[neuron]] = True
            # if neuron == 1:
            #     overlapping_spike_bool = find_overlapping_spike_bool(self.actual_IDs[neuron], self.actual_IDs[neuron-1], overlap_tol=162)
            #     self.actual_IDs[neuron] = self.actual_IDs[neuron][~overlapping_spike_bool]
            #     spiketrain[:] = False
            #     spiketrain[self.actual_IDs[neuron]] = True

            if self.correlate1_2 and neuron == 1:
                print("!!! MAKING UNIT 2 CORRELATE WITH UNIT 1 !!!")
                n_correlated_spikes = self.actual_IDs[neuron].shape[0] // 5
                select_inds0 = np.random.choice(self.actual_IDs[neuron-1].shape[0], n_correlated_spikes, replace=False)
                select_inds1 = np.random.choice(self.actual_IDs[neuron].shape[0], n_correlated_spikes, replace=False)
                self.actual_IDs[neuron][select_inds1] = self.actual_IDs[neuron-1][select_inds0] + np.random.randint(0, 10, n_correlated_spikes)
                self.actual_IDs[neuron].sort()
                overlapping_spike_bool = find_overlapping_spike_bool(self.actual_IDs[neuron], self.actual_IDs[neuron], overlap_tol=int(1.5e-3 * 40000), except_equal=True)
                self.actual_IDs[neuron] = self.actual_IDs[neuron][~overlapping_spike_bool]
                self.actual_IDs[neuron] = np.unique(self.actual_IDs[neuron])
                # Spike train is used for actual convolution so reset here
                spiketrain[:] = False
                spiketrain[self.actual_IDs[neuron]] = True

            for chan in range(0, self.num_channels):
                # Apply spike train to every channel this neuron is present on
                convolve_kernel = chan_scaling_factors[neuron][chan] * self.neuron_templates[template_inds[neuron], :]
                # Filter the kernel to match the noise band
                convolve_kernel = signal.filtfilt(b_filt, a_filt, convolve_kernel, axis=0, padlen=None)
                convolve_kernel = convolve_kernel.astype(self.voltage_dtype)
                self.actual_templates[neuron] = np.hstack((self.actual_templates[neuron], convolve_kernel.reshape(1, -1)))
                if chan_scaling_factors[neuron][chan] <= 0:
                    continue
                self.voltage_array[chan, :] += (signal.fftconvolve(spiketrain, convolve_kernel, mode='same')).astype(self.voltage_dtype)

    def gen_test_dataset_from_spikes(self, spiketrain, template_inds, chan_scaling_factors):
        """ Generates the random voltage array and adds spikes at the times
        indicated by spiketrain, rather than randomly generating them here.
        Allows the user to specify a spike train other than the Poisson random
        generated by gen_test_dataset. spiketrain is a boolean index indicating
        spike occurance for each sample of voltage (duration * sampling rate).
        """

        low_cutoff = self.frequency_range[0] / (self.samples_per_second / 2)
        high_cutoff = self.frequency_range[1] / (self.samples_per_second / 2)
        b_filt, a_filt = signal.butter(1, [low_cutoff, high_cutoff], btype='band')

        # Reset neuron actual IDs for each neuron
        self.actual_IDs = [[] for neur in range(0, len(template_inds))]
        self.actual_templates = [np.zeros((1, 0), dtype=self.voltage_dtype) for neur in range(0, len(firing_rates))]
        if self.voltage_array is None:
            self.gen_noise_voltage_array()
        for neuron in range(0, len(template_inds)):
            # Generate one spike train for each neuron
            self.actual_IDs[neuron] = np.where(spiketrain[neuron])[0]
            for chan in range(0, self.num_channels):
                # Apply spike train to every channel this neuron is present on
                convolve_kernel = chan_scaling_factors[neuron][chan] * self.neuron_templates[template_inds[neuron], :]
                # Filter the kernel to match the noise band
                convolve_kernel = signal.filtfilt(b_filt, a_filt, convolve_kernel, axis=0, padlen=None)
                convolve_kernel = convolve_kernel.astype(self.voltage_dtype)
                self.actual_templates[neuron] = np.hstack((self.actual_templates[neuron], convolve_kernel.reshape(1, -1)))
                if chan_scaling_factors[neuron][chan] <= 0:
                    continue
                self.voltage_array[chan, :] += (signal.fftconvolve(spiketrain[neuron], convolve_kernel, mode='same')).astype(self.voltage_dtype)

    def sort_test_dataset(self, add_templates=None, sort_templates=None, rand_state=None, kwargs={}):

        self.sort_info = {'sigma_lower_bound': 2.0,
                        'sigma_template_ci': 4.0,
                        'get_adjusted_clips': False,
                        'max_gpu_memory': None,
                        'max_shift_inds': None,
                        'sigma': 4.}
        for key in kwargs:
            self.sort_info[key] = kwargs[key]

        # Subtract 1 from pre_time since clip width assumes time zero center
        pre_time = ((self.neuron_templates.shape[1] - 1) // 2) / self.samples_per_second
        post_time = (self.neuron_templates.shape[1] - (self.neuron_templates.shape[1] // 2)) / self.samples_per_second
        # These sort info values must match voltage and templates and cannot be
        # changed by input
        self.sort_info.update({'n_channels': self.num_channels,
                     'n_samples_per_chan': self.neuron_templates.shape[1],
                     'clip_width': [pre_time, post_time],
                     'sampling_rate': self.samples_per_second,
                     'n_samples': self.voltage_array.shape[1]
                     })
        bp_chan_win, _ = time_window_to_samples(self.sort_info['clip_width'], self.sort_info['sampling_rate'])
        if self.sort_info['max_shift_inds'] is None:
            self.sort_info['max_shift_inds'] = self.sort_info['n_samples_per_chan'] - 1

        bp_templates = np.vstack(self.actual_templates)
        if sort_templates is not None:
            bp_templates = bp_templates[sort_templates, :]
            if bp_templates.ndim == 1:
                bp_templates = bp_templates.reshape(1, -1)

        self.full_chan_covariance_mats = noise_covariance_parallel(
                            self.voltage_array, [bp_chan_win[0], bp_chan_win[1]],
                            10000, rand_state=rand_state)

        # Need to find threshold crossings here since we are not running
        # clustering algorithm where it is usually done.
        self.sort_info['n_threshold_crossings'] = np.zeros(self.voltage_array.shape[0])
        chan_thresholds = median_threshold(self.voltage_array, self.sort_info['sigma'])
        for chan in range(0, self.voltage_array.shape[0]):
            _, self.sort_info['n_threshold_crossings'][chan] = spikesorting_fullpursuit.parallel.segment_parallel.identify_threshold_crossings(
                            self.voltage_array[chan, :], self.sort_info, chan_thresholds[chan])

        self.separability_metrics = neuron_separability.compute_separability_metrics(
                                                bp_templates,
                                                self.full_chan_covariance_mats,
                                                self.sort_info)
        noisy_templates = neuron_separability.find_noisy_templates(
                                        self.separability_metrics, self.sort_info)
        self.separability_metrics, noisy_templates = neuron_separability.del_noise_templates_and_threshold(
                                        self.separability_metrics,
                                        self.sort_info,
                                        noisy_templates)
        if self.separability_metrics['templates'].size == 0:
            print("All templates were removed")
            self.binary_pursuit_results = {'spike_indices': [],
                                           'neuron_labels': []}
            return
        crossings, neuron_labels, bp_bool, _ = binary_pursuit_parallel.binary_pursuit(
                        self.voltage_array, self.voltage_dtype,
                        self.sort_info, self.separability_metrics,
                        n_max_shift_inds=self.sort_info['max_shift_inds'],
                        kernels_path=None, max_gpu_memory=self.sort_info['max_gpu_memory'])

        self.binary_pursuit_results = {'spike_indices': [],
                                       'label': []}
        # Map the binary pursuit spike times to neuron labels that should match actual_IDs
        for n_num in range(0, self.separability_metrics['templates'].shape[0]):
            n_num_spike_indices = crossings[neuron_labels == n_num]
            index_order = np.argsort(n_num_spike_indices)
            n_num_spike_indices = n_num_spike_indices[index_order]
            self.binary_pursuit_results['label'].append(n_num)
            self.binary_pursuit_results['spike_indices'].append(n_num_spike_indices)
