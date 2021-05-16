import numpy as np
import os
import platform as sys_platform
import re
import time
from spikesorting_fullpursuit.sort import reorder_labels
from spikesorting_fullpursuit import segment, neuron_separability
from spikesorting_fullpursuit.parallel import segment_parallel
from scipy.signal import fftconvolve



def get_zero_phase_kernel(x, x_center):
    """ Zero pads the 1D kernel x, so that it is aligned with the current element
        of x located at x_center.  This ensures that convolution with the kernel
        x will be zero phase with respect to x_center.
    """

    kernel_offset = x.size - 2 * x_center - 1 # -1 To center ON the x_center index
    kernel_size = np.abs(kernel_offset) + x.size
    if kernel_size // 2 == 0: # Kernel must be odd
        kernel_size -= 1
        kernel_offset -= 1
    kernel = np.zeros(kernel_size)
    if kernel_offset > 0:
        kernel_slice = slice(kernel_offset, kernel.size)
    elif kernel_offset < 0:
        kernel_slice = slice(0, kernel.size + kernel_offset)
    else:
        kernel_slice = slice(0, kernel.size)
    kernel[kernel_slice] = x

    return kernel


def compute_shift_indices(templates, samples_per_chan, n_chans):
    """ Performs the max likelihood computation for various shifts of all pairwise
    templates in the absence of noise to determine the range of indices at which
    the algorithm will fail to correctly identify a spike label or index. """
    # Output is a flat matrix lookup for pairwise pre and post shift indices
    template_pre_inds = np.zeros(templates.shape[0] * templates.shape[0], dtype=np.int32)
    template_post_inds = np.zeros(templates.shape[0] * templates.shape[0], dtype=np.int32)

    # This will be vector of pseudo voltage to convolve summed templates
    sum_voltage = np.zeros((n_chans, 3 * samples_per_chan))
    L_phase = np.zeros(3*samples_per_chan)

    # Reference index for max value of each template. Get empirically because
    # nonlinear phase shift is confusing
    ref_inds = np.zeros(templates.shape[0], dtype=np.int64)
    for t in range(0, templates.shape[0]):
        L_phase[:] = 0
        for chan in range(0, n_chans):
            sum_voltage[chan, samples_per_chan:2*samples_per_chan] = templates[t, :][chan*samples_per_chan:(chan+1)*samples_per_chan]
            L_phase += np.convolve(sum_voltage[chan, :], templates[t, :][chan*samples_per_chan:(chan+1)*samples_per_chan], mode='same')
        ref_inds[t] = np.argmax(L_phase)

    # And corresponding likelihood functions
    L_fixed = np.zeros(3*samples_per_chan)
    L_shift = np.zeros(3*samples_per_chan)
    max_shift = int(samples_per_chan // 2)
    # NOTE: Need to add 1 to size of failed_assignments to include 0 shift at failed_assignments[max_shift]
    failed_assignments = np.zeros(2*max_shift+1, dtype=np.bool)
    for f in range(0, templates.shape[0]):
        fixed_t = templates[f, :]
        fixed_t_ss = .5 * np.sum(fixed_t ** 2)
        fixed_ind = ref_inds[f]
        for s in range(f, templates.shape[0]):
            shift_t = templates[s, :]
            shift_t_ss = .5 * np.sum(shift_t ** 2)
            failed_assignments[:] = False
            for shift in range(-max_shift, max_shift+1):
                # Reset variables for this shift
                shift_ind = ref_inds[s] + shift
                L_fixed[:] = 0
                L_shift[:] = 0
                sum_voltage[:] = 0

                # Compute template convolution aspect of likelihood function over channels
                for chan in range(0, n_chans):
                    # Create pseudo voltage containing both templates offset by shift indices
                    sum_voltage[chan, samples_per_chan:2*samples_per_chan] = fixed_t[chan*samples_per_chan:(chan+1)*samples_per_chan]
                    if shift < 0:
                        sum_voltage[chan, samples_per_chan+shift:2*samples_per_chan+shift] += shift_t[chan*samples_per_chan:(chan+1)*samples_per_chan]
                    else:
                        sum_voltage[chan, samples_per_chan+shift:2*samples_per_chan+shift] += shift_t[chan*samples_per_chan:(chan+1)*samples_per_chan]
                    L_fixed += np.convolve(sum_voltage[chan, :], fixed_t[chan*samples_per_chan:(chan+1)*samples_per_chan], mode='same')
                    L_shift += np.convolve(sum_voltage[chan, :], shift_t[chan*samples_per_chan:(chan+1)*samples_per_chan], mode='same')

                # Add template aspect of likelihood function and find peak indices
                L_fixed -= fixed_t_ss
                L_shift -= shift_t_ss
                fixed_max_ind = np.argmax(L_fixed)
                shift_max_ind = np.argmax(L_shift)

                if f == s:
                    if fixed_max_ind != fixed_ind and shift_max_ind != shift_ind:
                        # This would be an incorrect assignment and/or index
                        failed_assignments[shift + max_shift] = True
                elif L_fixed[fixed_max_ind] > L_shift[shift_max_ind]:
                    # Would add fixed template
                    if fixed_max_ind != fixed_ind:
                        # This would be an incorrect assignment and/or index
                        failed_assignments[shift + max_shift] = True
                else:
                    # Would add shift template
                    if shift_max_ind != shift_ind:
                        # This would be an incorrect assignment and/or index
                        failed_assignments[shift + max_shift] = True

            if np.count_nonzero(failed_assignments) == 0:
                template_pre_inds[f*templates.shape[0] + s] = 0
                template_post_inds[f*templates.shape[0] + s] = 0
                # Will be the same for reflective pairing
                template_pre_inds[s*templates.shape[0] + f] = 0
                template_post_inds[s*templates.shape[0] + f] = 0
            else:
                template_pre_inds[f*templates.shape[0] + s] = np.argmax(failed_assignments) - max_shift
                # NOTE: Subtracting 1 will give inclusive indices. Not subtracting 1 gives the slice range.
                template_post_inds[f*templates.shape[0] + s] = failed_assignments.shape[0] - np.argmax(failed_assignments[-1::-1]) - max_shift

                # For the same pair in opposite order, these indices are REVERSED
                template_pre_inds[s*templates.shape[0] + f] = -1 * template_post_inds[f*templates.shape[0] + s] + 1 # Add 1 to undo slice
                template_post_inds[s*templates.shape[0] + f] = -1 * template_pre_inds[f*templates.shape[0] + s] + 1

    return template_pre_inds, template_post_inds


def binary_pursuit(voltage, v_dtype, sort_info,
                   separability_metrics, n_max_shift_inds=None, kernels_path=None,
                   max_gpu_memory=None):
    """
    	binary_pursuit_opencl(voltage, crossings, labels, clips)

    Uses OpenCL on a GPU to identify spikes that overlap with already identified
    spikes. This also will identify spikes that are less than the threshold.
    The algorithm is based, loosely, on work by Johnathan Pillow (Princeton)
    published in Plos One (2013): A Model-Based Spike Sorting Algorithm for
    Removing Correlation Artifacts in Multi-Neuron Recordings.

    Computes the mean squared error (MSE) for every spike at every instance in
    time in the voltage trace and asks whether the addition of a particular spike
    template at each point in time would reduce the overall MSE. This function essentially
    separates overlapping spikes by finding the spike times for each template that
    minimize the squared error.

    The output of this function is a new set of crossing times, labels, and clips.
    The clips have all other spikes removed. The clips and binary pursuit are all
    done in np.float32 precision. Output clips are cast to templates.dtype
    at the end for output.

    A couple of notes regarding the OpenCL implementation of the binary pursuit algorithm.
     1. The labels passed into the both the residual computation and the binary pursuit
        OpenCL kernels *MUST* refer to their row in the templates array, in standard C
        (base 0) notation. That is, the first row of templates should be assigned a label of
        0, the second row should have a label of 1 and so on.
     2. The spike indices passed into the kernels must refer to the index of the onset of the
        template. If the index refers to an alignment point in the middle of the clip (e.g.,
        0.3ms after the onset of the clip), the spike indices must be shifted back to refer
        to the start of the clip before calling either kernel.
     4. The templates passed into the OpenCL kernels must be a vector not a matrix. Initially,
        we compute the templates for each label as MxN vector (M being the number of units, N
        being the number of points in each template). Before calling the CL kernels, this must
        be converted to 1x(M*N) vector where the first N points refer to the first template.

    """
    ############################################################################
    # Must reserve all references to pyopencl to be inside this function.
    # Otherwise importing by main calling function (spikesorting_parallel)
    # instantiates pyopencl in the host process and blocks the children
    import pyopencl as cl
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    ############################################################################
    # Ease of use variables
    n_chans = np.uint32(voltage.shape[0])
    n_samples = voltage.shape[1]
    sampling_rate = sort_info['sampling_rate']
    chan_win, _ = segment_parallel.time_window_to_samples(sort_info['clip_width'], sampling_rate)
    template_samples_per_chan = sort_info['n_samples_per_chan']

    templates = separability_metrics['templates']
    # Templates must be a 2D float32
    if templates.ndim == 1:
        templates = np.reshape(templates, (1, -1))
    templates = np.float32(templates)
    # Reshape our templates so that instead of being an MxN array, this
    # becomes a 1x(M*N) vector. The first template should be in the first N points
    templates_vector = templates.reshape(templates.size).astype(np.float32)
    # Make template labels, numbered from 0-n
    template_labels = [x for x in range(0, templates.shape[0])]

    # Adjust event_indices so that they refer to the start of the clip to be
    # subtracted.
    clip_init_samples = int(np.abs(chan_win[0]))
    template_labels = np.uint32(template_labels)

    # Get info needed for binary pursuit (see note below). Do this up here so it
    # is available for estimating GPU memory usage.
    # Multiply by -0.5 to accomodate simplified delta likelihood function
    template_sum_squared = np.float32(-0.5 * separability_metrics['template_SS'])
    template_sum_squared_by_channel = np.float32(separability_metrics['template_SS_by_chan'].flatten(order='C'))
    neuron_lower_thresholds = np.float32(separability_metrics['neuron_lower_thresholds'])
    # neuron_upper_thresholds = np.float32(separability_metrics['neuron_upper_thresholds'])

    # Load OpenCL code from a file (stored as a string)
    if kernels_path is None:
        current_path = os.path.realpath(__file__)
        if sys_platform.system() == 'Windows':
            kernels_path = current_path.split('\\')
            # slice at -2 if parallel folder is same level as kernels folder
            kernels_path = [x + '\\' for x in kernels_path[0:-2]]
            kernels_path = ''.join(kernels_path) + 'kernels\\binary_pursuit.cl'
        else:
            kernels_path = current_path.split('/')
            # slice at -2 if parallel folder is same level as kernels folder
            kernels_path = [x + '/' for x in kernels_path[0:-2]]
            kernels_path = ''.join(kernels_path) + 'kernels/binary_pursuit.cl'
    with open(kernels_path, 'r') as fp:
        kernels = fp.read()

    # Search for a platform with a GPU device, and pick GPU with most global memory
    platforms = cl.get_platforms()
    device = None
    max_global_mem = 0
    for platform in platforms:
        devices = platform.get_devices(cl.device_type.GPU)
        if len(devices) > 0:
            for d in range(0, len(devices)):
                if devices[d].get_info(cl.device_info.GLOBAL_MEM_SIZE) > max_global_mem:
                    device = devices[d]
    if device is None:
        raise RuntimeError("No GPU available")

    # Create a context and build our program
    ctx = cl.Context([device])
    mf = cl.mem_flags
    with cl.CommandQueue(ctx) as queue:
        prg = cl.Program(ctx, kernels)
        prg.build() # David has a bunch of PYOPENCL_BUILD_OPTIONS used here

        # Get all of our kernels
        compute_residual_kernel = prg.compute_residual
        compute_full_likelihood_kernel = prg.compute_full_likelihood
        compute_template_maximum_likelihood_kernel = prg.compute_template_maximum_likelihood
        overlap_recheck_indices_kernel = prg.overlap_recheck_indices
        parse_overlap_recheck_indices_kernel = prg.parse_overlap_recheck_indices
        check_overlap_reassignments_kernel = prg.check_overlap_reassignments
        binary_pursuit_kernel = prg.binary_pursuit
        get_adjusted_clips_kernel = prg.get_adjusted_clips

        # Set local work size for both kernels to the max for device
        # resid_local_work_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        # pursuit_local_work_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        # Set local work size for both kernels to the max for device/kernel 'preferred size?'
        # This can be far faster and leave computer display performance better intact
        # than using the MAX_WORK_GROUP_SIZE.
        # This was definitely true on the NVIDIA windows 10 configuration
        resid_local_work_size = np.int64(compute_residual_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        likelihood_local_work_size = np.int64(compute_full_likelihood_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        max_likelihood_local_work_size = np.int64(compute_template_maximum_likelihood_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        overlap_local_work_size = np.int64(overlap_recheck_indices_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        parse_overlap_local_work_size = np.int64(parse_overlap_recheck_indices_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        check_overlap_local_work_size = np.int64(check_overlap_reassignments_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        pursuit_local_work_size = np.int64(binary_pursuit_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        adjusted_local_work_size = np.int64(get_adjusted_clips_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device))
        # Over enqueuing can clog up the GPU and cause a GPU timeout, resulting
        # in either a system crash or the OS terminating our operation
        # These are later used to compute very large numbers so convert to int64
        max_enqueue_resid = np.int64(resid_local_work_size * (device.max_compute_units))
        max_enqueue_likelihood = np.int64(likelihood_local_work_size * (device.max_compute_units))
        max_enqueue_max_likelihood = np.int64(max_likelihood_local_work_size * (device.max_compute_units))
        max_enqueue_overlap = np.int64(overlap_local_work_size * (device.max_compute_units))
        max_enqueue_parse_overlap = np.int64(parse_overlap_local_work_size * (device.max_compute_units))
        max_enqueue_check_overlap = np.int64(check_overlap_local_work_size * (device.max_compute_units))
        max_enqueue_pursuit = np.int64(pursuit_local_work_size * (device.max_compute_units))
        max_enqueue_adjusted = np.int64(adjusted_local_work_size * (device.max_compute_units))

        # Determine the segment size that we need to use to fit into the
        # memory on this GPU device.
        gpu_memory_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) # Total memory size in bytes
        # Save 50% or 1GB, whichever is less for the operating system to use
        # assuming this is not a headless system.
        gpu_memory_size = max(gpu_memory_size * 0.5, gpu_memory_size - (1024 * 1024 * 1024))
        if max_gpu_memory is not None:
            gpu_memory_size = min(gpu_memory_size, max_gpu_memory)
        # Windows 10 only allows 75% of the memory on the graphics card to be used,
        # the rest is reserved for WDDMv2. See discussion here:
        # https://www.nvidia.com/en-us/geforce/forums/geforce-graphics-cards/5/269554/windows-10wddm-grabbing-gpu-vram/
        regex_version = re.search('[0-9][0-9][.]', sys_platform.version())
        if (sys_platform.system() == 'Windows') and (float(regex_version.group()) >= 10.):
            gpu_memory_size = min(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 0.75,
                                  gpu_memory_size)
        assert gpu_memory_size > 0

        # Need to comute this stuff up here for GPU memory usage calculation
        if n_max_shift_inds is None or n_max_shift_inds < 0:
            # Default to half clip width
            n_max_shift_inds = template_samples_per_chan // 2
        # Must be uint32
        n_max_shift_inds = np.uint32(n_max_shift_inds)
        print("Maximum shift indices to check", n_max_shift_inds)
        # NOTE: Everything about the overlaps kernel numbers must be int64
        # or risk overflow
        items_per_index = np.int64((2 * n_max_shift_inds + 1) * (2 * n_max_shift_inds + 1) * templates.shape[0])
        # Round ceiling gives number of work groups needed per recheck index
        n_local_ID = np.int64(((items_per_index - 1) / overlap_local_work_size)+1)

        # Estimate the number of bytes that 1 second of data take up
        # This is skipping the bias and template squared error buffers which are small
        constant_memory_usage = templates_vector.nbytes + template_labels.nbytes
        # Usage for voltage
        memory_usage_per_second = (n_chans * sampling_rate * np.dtype(np.float32).itemsize)
        # Add usage for likelihood functions
        memory_usage_per_second += (templates.shape[0] * sampling_rate * np.dtype(np.float32).itemsize)
        # Usage for new spike info storage buffers depends on samples divided by template size
        memory_usage_per_second += (sampling_rate / template_samples_per_chan) * \
                                    (np.dtype(np.uint32).itemsize + # additional indices
                                     np.dtype(np.uint32).itemsize + # additional labels
                                     np.dtype(np.uint32).itemsize + # best indices
                                     np.dtype(np.uint32).itemsize + # best labels
                                     np.dtype(np.float32).itemsize + # best likelihoods
                                     # n_local_ID / 3 because we can't add a spike at more than 1 in 3 windows at a time
                                     # NOTE: this should basically be worst case and could arbitrarily
                                     # be made smaller to avoid doing multiple chunks in most cases. Probably
                                     # not quite by a factor of 10, by 5 or 6 qualitatively should cover
                                     # most cases.
                                     (n_local_ID / 3) * np.dtype(np.float32).itemsize + # Overlap likelihoods
                                     (n_local_ID / 3) * np.dtype(np.uint32).itemsize ) # Overlap labels

        # Estimate how much data we can look at in each data 'chunk'
        num_seconds_per_chunk = (gpu_memory_size - constant_memory_usage) \
                                / (memory_usage_per_second)
        # Do not exceed the length of a buffer in bytes that can be addressed
        # for the single largest data buffer, the voltage
        # Note: there is also a max allocation size,
        # device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        device_address_bits = device.get_info(cl.device_info.ADDRESS_BITS)
        max_addressable_seconds = (((1 << device_address_bits) - 1)
                                    / (np.dtype(np.float32).itemsize * sampling_rate
                                     * n_chans))
        num_seconds_per_chunk = min(num_seconds_per_chunk, np.floor(max_addressable_seconds))
        # Convert from seconds to indices, the actual currency of this function's computations
        num_indices_per_chunk = int(np.floor(num_seconds_per_chunk * sampling_rate))

        if num_indices_per_chunk < 4 * template_samples_per_chan:
            raise ValueError("Cannot fit enough data on GPU to run binary pursuit. Decrease channels or get a new GPU.")

        # Set up our final outputs
        num_additional_spikes = np.zeros(1, dtype=np.uint32)

        # Set-up any buffers/lists that are not dependent on chunks
        secret_spike_indices = []
        secret_spike_labels = []
        secret_spike_bool = []
        adjusted_spike_clips = []

        template_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=templates_vector)
        template_sum_squared_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_sum_squared)
        template_sum_squared_by_channel_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_sum_squared_by_channel)

        # Determine our chunk onset indices, making sure that each new start
        # index overlaps the previous chunk by 3 template widths so that no
        # spikes are missed by binary pursuit
        # Note: Any spikes in the last template width of data are not checked
        chunk_onsets = []
        curr_onset = 0
        while ((curr_onset) < (n_samples - template_samples_per_chan)):
            chunk_onsets.append(curr_onset)
            curr_onset += num_indices_per_chunk - 3 * template_samples_per_chan
        print("Using ", len(chunk_onsets), " chunks for binary pursuit.", flush=True)

        # Loop over chunks
        for chunk_number, start_index in enumerate(chunk_onsets):
            stop_index = np.uint32(min(n_samples, start_index + num_indices_per_chunk))
            print("Starting chunk number", chunk_number, "from", start_index, "to", stop_index, "samples", flush=True)
            chunk_voltage = np.float32(voltage[:, start_index:stop_index])
            chunk_voltage_length = np.uint32(stop_index - start_index)
            # Reshape voltage over channels into a single 1D vector
            chunk_voltage = chunk_voltage.reshape(chunk_voltage.size)

            # We will find all spikes in binary pursuit so set chunk crossings to zero
            chunk_crossings = np.array([], dtype=np.uint32)
            chunk_labels = np.array([], dtype=np.uint32)

            # - Set up the compute_residual kernel -
            # Create our buffers on the graphics cards.
            # Essentially all we are doing is copying each of our arrays to the graphics card.
            voltage_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=chunk_voltage)
            full_likelihood_function_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(chunk_voltage_length * templates.shape[0], dtype=np.float32))

            # Set arguments that are the same every iteration
            compute_residual_kernel.set_arg(0, voltage_buffer) # location where chunk voltage is stored
            compute_residual_kernel.set_arg(1, chunk_voltage_length) # length of chunk voltage array
            compute_residual_kernel.set_arg(2, n_chans) # Number of neighboring channels
            compute_residual_kernel.set_arg(3, template_buffer) # location where our templates are stored
            compute_residual_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of neurons (`rows` in templates)
            compute_residual_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each channel of templates

            next_wait_event = None

            # Determine how many independent workers will work on binary pursuit
            # (Includes both template ML and binary pursuit kernels)
            # NOTE: As written, num_template_widths can't exceed uint32
            num_template_widths = int(np.ceil(chunk_voltage_length / template_samples_per_chan))
            # These are the work sizes that need computed before while loop below
            # then updated at the end of while loop
            total_work_size_likelihood = likelihood_local_work_size * np.int64(np.ceil((num_template_widths * templates.shape[0]) / likelihood_local_work_size))
            total_work_size_max_likelihood = max_likelihood_local_work_size * np.int64(np.ceil(num_template_widths / max_likelihood_local_work_size))
            total_work_size_pursuit = pursuit_local_work_size * np.int64(np.ceil(num_template_widths / pursuit_local_work_size))

            # Construct our buffers
            lower_thresholds_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=neuron_lower_thresholds)
            # upper_thresholds_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=neuron_upper_thresholds)
            num_additional_spikes_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=np.uint32)) # NOTE: Must be :rw for atomic to work
            additional_spike_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            additional_spike_labels_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            best_spike_likelihoods_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.float32))
            best_spike_labels_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            best_spike_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            window_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.arange(0, num_template_widths, 1, dtype=np.uint32))
            next_check_window_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint8))
            overlap_recheck_window_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint8))
            overlap_window_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.arange(0, num_template_widths, 1, dtype=np.uint32))
            overlap_best_spike_indices_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))
            overlap_best_spike_labels_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_template_widths, dtype=np.uint32))

            # This is separate storage for transferring data from next check, NOT A HOST (maybe could be)
            next_check_window = np.zeros(num_template_widths, dtype=np.uint8)
            overlap_recheck_window = np.zeros(num_template_widths, dtype=np.uint8)

            # Set input arguments for template maximum likelihood kernel
            compute_full_likelihood_kernel.set_arg(0, voltage_buffer) # Voltage buffer created previously
            compute_full_likelihood_kernel.set_arg(1, chunk_voltage_length) # Length of chunk voltage
            compute_full_likelihood_kernel.set_arg(2, n_chans) # number of neighboring channels
            compute_full_likelihood_kernel.set_arg(3, template_buffer) # Our template buffer, created previously
            compute_full_likelihood_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of unique neurons, M
            compute_full_likelihood_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            compute_full_likelihood_kernel.set_arg(6, template_sum_squared_buffer) # Sum of squared templated
            compute_full_likelihood_kernel.set_arg(7, window_indices_buffer) # Actual window indices to check
            compute_full_likelihood_kernel.set_arg(8, np.uint32(num_template_widths)) # Number of actual window indices to check
            compute_full_likelihood_kernel.set_arg(9, full_likelihood_function_buffer)

            # Set input arguments for template maximum likelihood kernel
            compute_template_maximum_likelihood_kernel.set_arg(0, chunk_voltage_length) # Length of chunk voltage
            compute_template_maximum_likelihood_kernel.set_arg(1, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            compute_template_maximum_likelihood_kernel.set_arg(2, np.uint32(0)) # Template number
            compute_template_maximum_likelihood_kernel.set_arg(3, window_indices_buffer) # Actual window indices to check
            compute_template_maximum_likelihood_kernel.set_arg(4, np.uint32(num_template_widths)) # Number of actual window indices to check
            compute_template_maximum_likelihood_kernel.set_arg(5, best_spike_indices_buffer) # Storage for peak likelihood index
            compute_template_maximum_likelihood_kernel.set_arg(6, best_spike_labels_buffer) # Storage for peak likelihood label
            compute_template_maximum_likelihood_kernel.set_arg(7, best_spike_likelihoods_buffer) # Storage for peak likelihood value
            compute_template_maximum_likelihood_kernel.set_arg(8, next_check_window_buffer) # Binary vector indicating whether a window at its index needs checked on next iteration of binary_pursuit kernel
            compute_template_maximum_likelihood_kernel.set_arg(9, overlap_recheck_window_buffer)
            compute_template_maximum_likelihood_kernel.set_arg(10, overlap_best_spike_indices_buffer) # Storage for new best overlap indices
            compute_template_maximum_likelihood_kernel.set_arg(11, overlap_best_spike_labels_buffer) # Storage for new best overlap indices
            compute_template_maximum_likelihood_kernel.set_arg(12, full_likelihood_function_buffer)
            compute_template_maximum_likelihood_kernel.set_arg(13, lower_thresholds_buffer)

            # Set input arguments for overlap recheck kernel
            overlap_recheck_indices_kernel.set_arg(0, voltage_buffer) # Voltage buffer created previously
            overlap_recheck_indices_kernel.set_arg(1, chunk_voltage_length) # Length of chunk voltage
            overlap_recheck_indices_kernel.set_arg(2, n_chans) # number of neighboring channels
            overlap_recheck_indices_kernel.set_arg(3, template_buffer) # Our template buffer, created previously
            overlap_recheck_indices_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of unique neurons, M
            overlap_recheck_indices_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            overlap_recheck_indices_kernel.set_arg(6, overlap_window_indices_buffer) # Actual window indices to check
            overlap_recheck_indices_kernel.set_arg(7, np.uint32(num_template_widths)) # Number of actual window indices to check
            overlap_recheck_indices_kernel.set_arg(8, best_spike_indices_buffer) # Storage for peak likelihood index
            overlap_recheck_indices_kernel.set_arg(9, best_spike_labels_buffer) # Storage for peak likelihood label
            overlap_recheck_indices_kernel.set_arg(10, full_likelihood_function_buffer)
            overlap_recheck_indices_kernel.set_arg(11, n_max_shift_inds)
            # Construct a local buffer (float32 * local_work_size)
            overlap_local_likelihood_buffer = cl.LocalMemory(4 * overlap_local_work_size)
            overlap_recheck_indices_kernel.set_arg(12, overlap_local_likelihood_buffer)
            # Construct a local buffer (uint32 * local_work_size)
            overlap_local_ids__buffer = cl.LocalMemory(4 * overlap_local_work_size)
            overlap_recheck_indices_kernel.set_arg(13, overlap_local_ids__buffer)

            # Set input arguments for parse overlap recheck kernel
            parse_overlap_recheck_indices_kernel.set_arg(0, chunk_voltage_length) # Length of chunk voltage
            parse_overlap_recheck_indices_kernel.set_arg(1, np.uint32(templates.shape[0])) # Number of unique neurons, M
            parse_overlap_recheck_indices_kernel.set_arg(2, overlap_window_indices_buffer) # Actual window indices to check
            parse_overlap_recheck_indices_kernel.set_arg(3, np.uint32(num_template_widths)) # Number of actual window indices to check
            parse_overlap_recheck_indices_kernel.set_arg(4, best_spike_indices_buffer) # Storage for peak likelihood index
            parse_overlap_recheck_indices_kernel.set_arg(5, best_spike_labels_buffer) # Storage for peak likelihood label
            parse_overlap_recheck_indices_kernel.set_arg(6, best_spike_likelihoods_buffer) # Storage for peak likelihood value
            parse_overlap_recheck_indices_kernel.set_arg(7, overlap_best_spike_indices_buffer) # Storage for new best overlap indices
            parse_overlap_recheck_indices_kernel.set_arg(8, overlap_best_spike_labels_buffer) # Storage for new best overlap indices
            parse_overlap_recheck_indices_kernel.set_arg(9, full_likelihood_function_buffer)
            parse_overlap_recheck_indices_kernel.set_arg(10, n_max_shift_inds)
            parse_overlap_recheck_indices_kernel.set_arg(11, lower_thresholds_buffer)

            check_overlap_reassignments_kernel.set_arg(0, chunk_voltage_length) # Length of chunk voltage
            check_overlap_reassignments_kernel.set_arg(1, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            check_overlap_reassignments_kernel.set_arg(2, overlap_window_indices_buffer) # Actual window indices to check
            check_overlap_reassignments_kernel.set_arg(3, np.uint32(num_template_widths)) # Number of actual window indices to check
            check_overlap_reassignments_kernel.set_arg(4, best_spike_indices_buffer) # Storage for peak likelihood index
            check_overlap_reassignments_kernel.set_arg(5, best_spike_labels_buffer) # Storage for peak likelihood label
            check_overlap_reassignments_kernel.set_arg(6, next_check_window_buffer) # Binary vector indicating whether a window at its index needs checked on next iteration of binary_pursuit kernel
            check_overlap_reassignments_kernel.set_arg(7, overlap_best_spike_indices_buffer) # Storage for new best overlap indices
            check_overlap_reassignments_kernel.set_arg(8, overlap_best_spike_labels_buffer) # Storage for new best overlap indices
            check_overlap_reassignments_kernel.set_arg(9, overlap_recheck_window_buffer)

            # Construct a local buffer (unsigned int * local_work_size)
            local_buffer = cl.LocalMemory(4 * pursuit_local_work_size)

            # Set input arguments for binary pursuit kernel
            binary_pursuit_kernel.set_arg(0, chunk_voltage_length) # Length of chunk voltage
            binary_pursuit_kernel.set_arg(1, n_chans) # number of neighboring channels
            binary_pursuit_kernel.set_arg(2, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            binary_pursuit_kernel.set_arg(3, lower_thresholds_buffer) # Bias
            binary_pursuit_kernel.set_arg(4, window_indices_buffer) # Actual window indices to check
            binary_pursuit_kernel.set_arg(5, np.uint32(num_template_widths)) # Number of actual window indices to check
            binary_pursuit_kernel.set_arg(6, best_spike_indices_buffer) # Storage for peak likelihood index
            binary_pursuit_kernel.set_arg(7, best_spike_labels_buffer) # Storage for peak likelihood label
            binary_pursuit_kernel.set_arg(8, best_spike_likelihoods_buffer) # Storage for peak likelihood value
            binary_pursuit_kernel.set_arg(9, local_buffer) # Local buffer
            binary_pursuit_kernel.set_arg(10, num_additional_spikes_buffer) # Output, total number of additional spikes
            binary_pursuit_kernel.set_arg(11, additional_spike_indices_buffer) # Additional spike indices
            binary_pursuit_kernel.set_arg(12, additional_spike_labels_buffer) # Additional spike labels
            binary_pursuit_kernel.set_arg(13, overlap_recheck_window_buffer)
            binary_pursuit_kernel.set_arg(14, next_check_window_buffer) # Binary vector indicating whether a window at its index needs checked on next iteration of binary_pursuit kernel

            # Run the kernel until num_additional_spikes is zero
            chunk_total_additional_spikes = 0
            chunk_total_additional_spike_indices = []
            chunk_total_additional_spike_labels = []
            n_loops = 0
            print("pursuit total size is", total_work_size_pursuit, "local size is", pursuit_local_work_size, "with max enqueue", max_enqueue_pursuit, "chose n to enqueue", min(total_work_size_pursuit, max_enqueue_pursuit))
            print("overlaps local size is", overlap_local_work_size, "with max enqueue", max_enqueue_overlap)
            print("Looking in", num_template_widths, "windows")
            while True:
                n_loops += 1
                if n_loops % 1 == 0 and n_loops > 1:
                    print("Starting loop", n_loops, "for this chunk")
                    print("Next round has", new_window_indices.shape[0], "windows to check")



                n_to_enqueue = total_work_size_likelihood #np.amin([total_work_size_likelihood, max_enqueue_likelihood])
                for enqueue_step in np.arange(np.int64(0), total_work_size_likelihood, n_to_enqueue, dtype=np.int64):
                    # Avoid enqueueing potentially billions of workers to do nothing
                    if enqueue_step + n_to_enqueue > total_work_size_likelihood:
                        n_to_enqueue = likelihood_local_work_size * np.int64(np.ceil((total_work_size_likelihood - enqueue_step) / likelihood_local_work_size))
                    temp_ml_event = cl.enqueue_nd_range_kernel(queue,
                                          compute_full_likelihood_kernel,
                                          (n_to_enqueue, ), (likelihood_local_work_size, ),
                                          global_work_offset=(enqueue_step, ),
                                          wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [temp_ml_event]

                # n_to_enqueue = np.amin([total_work_size_max_likelihood, max_enqueue_pursuit])
                n_to_enqueue = total_work_size_max_likelihood
                for template_index in range(0, templates.shape[0]):
                    # Run one template at a time to avoid race
                    compute_template_maximum_likelihood_kernel.set_arg(2, np.uint32(template_index)) # Template number
                    for enqueue_step in np.arange(np.int64(0), total_work_size_max_likelihood, n_to_enqueue, dtype=np.int64):
                        temp_ml_event = cl.enqueue_nd_range_kernel(queue,
                                              compute_template_maximum_likelihood_kernel,
                                              (n_to_enqueue, ), (max_likelihood_local_work_size, ),
                                              global_work_offset=(enqueue_step, ),
                                              wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [temp_ml_event]


                # Read out the data from overlap_recheck_window_buffer
                next_wait_event = [cl.enqueue_copy(queue, overlap_recheck_window, overlap_recheck_window_buffer, wait_for=next_wait_event)]
                queue.finish() # Needs to finish copy before deciding indices
                # Use overlap_recheck_window data to determine window indices for overlap recheck
                overlap_window_indices = np.uint32(np.nonzero(overlap_recheck_window)[0])
                if overlap_window_indices.shape[0] > 0 and sort_info['do_overlap_recheck']:
                    # Still more flagged spikes to check
                    # Copy the overlap window indices to the overlap indices buffer
                    next_wait_event = [cl.enqueue_copy(queue, overlap_window_indices_buffer, overlap_window_indices, wait_for=next_wait_event)]
                    queue.finish() # Needs to finish copy before checking indices

                    # NOTE: Everything about the overlaps kernel numbers must be int64
                    # or risk overflow
                    # Reset number of indices to check for overlap recheck kernel
                    # Number of leftover workers for each index
                    n_local_ID_leftover = np.int64((overlap_local_work_size * n_local_ID) % items_per_index)
                    # Total work is then indices, times number per index, plus the leftovers from work groups
                    total_work_size_overlap = np.int64(overlap_window_indices.shape[0] * items_per_index + overlap_window_indices.shape[0] * n_local_ID_leftover)
                    # Round to nearest work group size
                    total_work_size_overlap = overlap_local_work_size * np.int64(np.ceil(total_work_size_overlap / overlap_local_work_size))
                    print("Checking", total_work_size_overlap, "overlap spike shifts")

                    if total_work_size_overlap > (1 << device_address_bits) - 1:
                        print("GPU has", device_address_bits, "address bits")
                        print("Current work size of", total_work_size_overlap, "will OVERFLOW a", device_address_bits, "bit graphics card.")
                        print("Decrease segment duration/number of channels in data file or get a new GPU.")

                    # Enqueues max_enqueue_overlap total spikes to be sorted.
                    # This yields a multiple such that at the end of each enqueue
                    # group, we could fully parse the spike results
                    # enqueue_by_spikes = n_local_ID * overlap_local_work_size * max_enqueue_overlap
                    # n_to_enqueue = np.amin([total_work_size_overlap, enqueue_by_spikes])
                    # Can't enque over uint 32 on Windows (maybe others) so this is max
                    enqueue_by_int32 = np.int64(overlap_local_work_size * np.ceil((2**32 - overlap_local_work_size) / overlap_local_work_size))
                    n_to_enqueue = np.amin([total_work_size_overlap, enqueue_by_int32])

                    overlap_recheck_indices_kernel.set_arg(7, np.uint32(overlap_window_indices.shape[0])) # Number of actual window indices to check
                    parse_overlap_recheck_indices_kernel.set_arg(3, np.uint32(overlap_window_indices.shape[0]))
                    num_work_groups = overlap_window_indices.shape[0] * n_local_ID

                    if n_loops == 1:
                        overlap_group_best_likelihood_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_work_groups, dtype=np.float32))
                        overlap_recheck_indices_kernel.set_arg(14, overlap_group_best_likelihood_buffer)
                        parse_overlap_recheck_indices_kernel.set_arg(12, overlap_group_best_likelihood_buffer)
                        overlap_group_best_work_id_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_work_groups, dtype=np.uint32))
                        overlap_recheck_indices_kernel.set_arg(15, overlap_group_best_work_id_buffer)
                        parse_overlap_recheck_indices_kernel.set_arg(13, overlap_group_best_work_id_buffer)
                        curr_overlaps_buffer_size = num_work_groups
                        print("Made buffers for a total of", num_work_groups, "work groups")

                    if num_work_groups > curr_overlaps_buffer_size:
                        # Number of work groups somehow went up, so we need new buffers to store everything
                        overlap_group_best_likelihood_buffer.release()
                        overlap_group_best_work_id_buffer.release()
                        overlap_group_best_likelihood_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_work_groups, dtype=np.float32))
                        overlap_recheck_indices_kernel.set_arg(14, overlap_group_best_likelihood_buffer)
                        parse_overlap_recheck_indices_kernel.set_arg(12, overlap_group_best_likelihood_buffer)
                        overlap_group_best_work_id_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(num_work_groups, dtype=np.uint32))
                        overlap_recheck_indices_kernel.set_arg(15, overlap_group_best_work_id_buffer)
                        parse_overlap_recheck_indices_kernel.set_arg(13, overlap_group_best_work_id_buffer)
                        curr_overlaps_buffer_size = num_work_groups
                        print("Made buffers for a total of", num_work_groups, "work groups")

                    for enqueue_step in np.arange(np.int64(0), total_work_size_overlap, n_to_enqueue, dtype=np.int64):
                        # Avoid enqueueing potentially billions of workers to do nothing
                        if enqueue_step + n_to_enqueue > total_work_size_overlap:
                            n_to_enqueue = overlap_local_work_size * np.int64(np.ceil((total_work_size_overlap - enqueue_step) / overlap_local_work_size))
                        overlap_event = cl.enqueue_nd_range_kernel(queue,
                                              overlap_recheck_indices_kernel,
                                              (n_to_enqueue, ), (overlap_local_work_size, ),
                                              global_work_offset=(enqueue_step, ),
                                              wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [overlap_event]

                    total_work_size_parse_overlap = parse_overlap_local_work_size * np.int64(np.ceil(overlap_window_indices.shape[0] / parse_overlap_local_work_size))
                    n_to_enqueue = total_work_size_parse_overlap #np.amin([total_work_size_parse_overlap, max_enqueue_parse_overlap])
                    for enqueue_step in np.arange(np.int64(0), total_work_size_parse_overlap, n_to_enqueue, dtype=np.int64):
                        overlap_event = cl.enqueue_nd_range_kernel(queue,
                                              parse_overlap_recheck_indices_kernel,
                                              (n_to_enqueue, ), (parse_overlap_local_work_size, ),
                                              global_work_offset=(enqueue_step, ),
                                              wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [overlap_event]

                    total_work_size_check_overlap = check_overlap_local_work_size * np.int64(np.ceil(overlap_window_indices.shape[0] / check_overlap_local_work_size))
                    n_to_enqueue = total_work_size_check_overlap #np.amin([total_work_size_check_overlap, max_enqueue_check_overlap])
                    check_overlap_reassignments_kernel.set_arg(3, np.uint32(overlap_window_indices.shape[0])) # Number of actual window indices to check
                    for enqueue_step in np.arange(np.int64(0), total_work_size_check_overlap, n_to_enqueue, dtype=np.int64):
                        overlap_event = cl.enqueue_nd_range_kernel(queue,
                                              check_overlap_reassignments_kernel,
                                              (n_to_enqueue, ), (check_overlap_local_work_size, ),
                                              global_work_offset=(enqueue_step, ),
                                              wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [overlap_event]

                n_to_enqueue = total_work_size_pursuit #np.amin([total_work_size_pursuit, max_enqueue_pursuit])
                for enqueue_step in np.arange(np.int64(0), total_work_size_pursuit, n_to_enqueue, dtype=np.int64):
                    pursuit_event = cl.enqueue_nd_range_kernel(queue,
                                          binary_pursuit_kernel,
                                          (n_to_enqueue, ), (pursuit_local_work_size, ),
                                          global_work_offset=(enqueue_step, ),
                                          wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [pursuit_event]

                cl.enqueue_copy(queue, num_additional_spikes, num_additional_spikes_buffer, wait_for=next_wait_event)
                print("Added", num_additional_spikes[0], "secret spikes", flush=True)
                if (num_additional_spikes[0] == 0):
                    break # Converged, no spikes added in last pass

                # Reset overlap_recheck_window for next pass and copy to GPU
                overlap_recheck_window[:] = 0
                next_wait_event = [cl.enqueue_copy(queue, overlap_recheck_window_buffer, overlap_recheck_window, wait_for=next_wait_event)]
                # Read out the data from next_check_window_buffer
                # Already waited for pursuit to finish above
                next_wait_event = [cl.enqueue_copy(queue, next_check_window, next_check_window_buffer, wait_for=None)]
                queue.finish() # Needs to finish copy before deciding indices
                # Use next_check_window data to determine window indices for next pass
                new_window_indices = np.uint32(np.nonzero(next_check_window)[0])
                # Copy the new window indices to the window indices buffer
                next_wait_event = [cl.enqueue_copy(queue, window_indices_buffer, new_window_indices, wait_for=next_wait_event)]
                # Reset number of indices to check for both kernels
                compute_full_likelihood_kernel.set_arg(8, np.uint32(new_window_indices.shape[0])) # Number of actual window indices to check
                compute_template_maximum_likelihood_kernel.set_arg(4, np.uint32(new_window_indices.shape[0])) # Number of actual window indices to check
                binary_pursuit_kernel.set_arg(5, np.uint32(new_window_indices.shape[0])) # Number of actual window indices to check
                # Reset number of kernels to run for next pass
                total_work_size_pursuit = pursuit_local_work_size * np.int64(np.ceil(new_window_indices.shape[0] / pursuit_local_work_size))
                total_work_size_likelihood = likelihood_local_work_size * np.int64(np.ceil((new_window_indices.shape[0] * templates.shape[0]) / likelihood_local_work_size))
                total_work_size_max_likelihood = max_likelihood_local_work_size * np.int64(np.ceil(new_window_indices.shape[0] / max_likelihood_local_work_size))
                # Reset next_check_window for next pass and copy to GPU
                next_check_window[:] = 0
                next_wait_event = [cl.enqueue_copy(queue, next_check_window_buffer, next_check_window, wait_for=next_wait_event)]

                # Read out and save all the new spikes we just found for this chunk
                additional_spike_indices = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                additional_spike_labels = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                next_wait_event = []
                # Already waited for pursuit to finish above
                next_wait_event.append(cl.enqueue_copy(queue, additional_spike_indices, additional_spike_indices_buffer, wait_for=None))
                next_wait_event.append(cl.enqueue_copy(queue, additional_spike_labels, additional_spike_labels_buffer, wait_for=None))
                chunk_total_additional_spike_indices.append(additional_spike_indices)
                chunk_total_additional_spike_labels.append(additional_spike_labels)

                # We have added additional spikes this pass, so we need to subtract them off
                # by calling the compute_residuals kernel. Most of the arguments are the
                # same as when we called it previously (e.g., voltage is the same buffer on the
                # GPU).
                compute_residual_kernel.set_arg(6, additional_spike_indices_buffer)
                compute_residual_kernel.set_arg(7, additional_spike_labels_buffer)
                compute_residual_kernel.set_arg(8, num_additional_spikes[0])

                total_work_size_resid = resid_local_work_size * np.int64(np.ceil(
                                            (num_additional_spikes[0]) / resid_local_work_size))

                n_to_enqueue = np.amin([total_work_size_resid, max_enqueue_resid])
                for enqueue_step in np.arange(np.int64(0), total_work_size_resid, n_to_enqueue, dtype=np.int64):
                    residual_event = cl.enqueue_nd_range_kernel(queue,
                                           compute_residual_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    queue.finish()
                    next_wait_event = [residual_event]
                # Ensure that num_additional_spikes is equal to zero for the next pass
                cl.enqueue_copy(queue, num_additional_spikes_buffer, np.zeros(1, dtype=np.uint32), wait_for=next_wait_event)
                chunk_total_additional_spikes += num_additional_spikes[0]
                num_additional_spikes[0] = 0
                queue.finish()

            additional_spike_indices_buffer.release()
            additional_spike_labels_buffer.release()
            lower_thresholds_buffer.release()
            best_spike_likelihoods_buffer.release()
            best_spike_labels_buffer.release()
            best_spike_indices_buffer.release()
            window_indices_buffer.release()
            next_check_window_buffer.release()
            overlap_recheck_window_buffer.release()
            overlap_window_indices_buffer.release()
            overlap_best_spike_indices_buffer.release()
            overlap_best_spike_labels_buffer.release()
            if sort_info['do_overlap_recheck']:
                overlap_group_best_likelihood_buffer.release()
                overlap_group_best_work_id_buffer.release()
            full_likelihood_function_buffer.release()
            queue.finish()

            # Read out the adjusted spikes here before releasing
            # the residual voltage. Only do this if there are spikes to get clips of
            if (chunk_total_additional_spikes + chunk_crossings.shape[0]) > 0:
                if chunk_total_additional_spikes == 0:
                    all_chunk_crossings = chunk_crossings
                    all_chunk_labels = chunk_labels
                elif chunk_crossings.shape[0] == 0:
                    all_chunk_crossings = np.hstack(chunk_total_additional_spike_indices)
                    all_chunk_labels = np.hstack(chunk_total_additional_spike_labels)
                else:
                    all_chunk_crossings = np.hstack((np.hstack(chunk_total_additional_spike_indices), chunk_crossings))
                    all_chunk_labels = np.hstack((np.hstack(chunk_total_additional_spike_labels), chunk_labels))

                print("Found", chunk_total_additional_spikes, "secret spikes this chunk.", flush=True)

                if sort_info['get_adjusted_clips']:
                    print("Getting adjusted clips for", all_chunk_crossings.shape[0], "total spikes this chunk", flush=True)

                    all_adjusted_clips = np.zeros((chunk_total_additional_spikes + chunk_crossings.shape[0]) * templates.shape[1], dtype=np.float32)
                    all_chunk_crossings_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_crossings)
                    all_chunk_labels_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_labels)
                    all_adjusted_clips_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=all_adjusted_clips)

                    get_adjusted_clips_kernel.set_arg(0, voltage_buffer)
                    get_adjusted_clips_kernel.set_arg(1, chunk_voltage_length)
                    get_adjusted_clips_kernel.set_arg(2, n_chans)
                    get_adjusted_clips_kernel.set_arg(3, template_buffer)
                    get_adjusted_clips_kernel.set_arg(4, np.uint32(templates.shape[0]))
                    get_adjusted_clips_kernel.set_arg(5, np.uint32(template_samples_per_chan))
                    get_adjusted_clips_kernel.set_arg(6, all_chunk_crossings_buffer)
                    get_adjusted_clips_kernel.set_arg(7, all_chunk_labels_buffer)
                    get_adjusted_clips_kernel.set_arg(8, np.uint32(all_chunk_crossings.shape[0]))
                    get_adjusted_clips_kernel.set_arg(9, all_adjusted_clips_buffer)

                    # Run the kernel using same sizes as for resid kernel
                    total_work_size_adjusted = adjusted_local_work_size * np.int64(np.ceil(
                                                (all_chunk_crossings.shape[0]) / adjusted_local_work_size))
                    clip_events = []
                    n_to_enqueue = np.amin([total_work_size_adjusted, max_enqueue_adjusted])
                    print("Getting adjusted clips", flush=True)
                    for enqueue_step in np.arange(np.int64(0), total_work_size_adjusted, n_to_enqueue, dtype=np.int64):
                        clip_event = cl.enqueue_nd_range_kernel(queue,
                                               get_adjusted_clips_kernel,
                                               (n_to_enqueue, ), (adjusted_local_work_size, ),
                                               global_work_offset=(enqueue_step, ),
                                               wait_for=next_wait_event)
                        queue.finish()
                        next_wait_event = [clip_event]
                    cl.enqueue_copy(queue, all_adjusted_clips, all_adjusted_clips_buffer, wait_for=clip_events)
                    queue.finish()
                    all_chunk_crossings_buffer.release()
                    all_chunk_labels_buffer.release()
                    all_adjusted_clips_buffer.release()
                    all_adjusted_clips = np.reshape(all_adjusted_clips, (all_chunk_crossings.shape[0], templates.shape[1]))
                else:
                    all_adjusted_clips = []

                # Save all chunk data and free spike dependent memory
                secret_spike_indices.append(all_chunk_crossings + start_index)
                secret_spike_labels.append(all_chunk_labels)
                adjusted_spike_clips.append(all_adjusted_clips)
                all_chunk_bool = np.zeros(all_chunk_crossings.shape[0], dtype=np.bool)
                all_chunk_bool[0:chunk_total_additional_spikes] = True
                secret_spike_bool.append(all_chunk_bool)

            # Release buffers before going on to next chunk
            voltage_buffer.release()
            num_additional_spikes_buffer.release()

        template_buffer.release()
        template_sum_squared_buffer.release()
        template_sum_squared_by_channel_buffer.release()

    if len(secret_spike_indices) > 0:
        event_indices = np.int64(np.hstack(secret_spike_indices))
        neuron_labels = np.int64(np.hstack(secret_spike_labels))
        adjusted_clips = (np.vstack(adjusted_spike_clips)).astype(v_dtype)
        binary_pursuit_spike_bool = np.hstack(secret_spike_bool)
        # Realign events with center of spike
        event_indices += clip_init_samples
        print("Found a total of", np.count_nonzero(binary_pursuit_spike_bool), "secret spikes", flush=True)
    else:
        # No spikes found and findall == True
        event_indices, neuron_labels, binary_pursuit_spike_bool, adjusted_clips = [], [], [], []
        print("Found a total of ZERO secret spikes", flush=True)

    # Separability metrics uses number of spikes assigned to each unit
    neuron_separability.add_n_spikes(separability_metrics, neuron_labels)

    return event_indices, neuron_labels, binary_pursuit_spike_bool, adjusted_clips
