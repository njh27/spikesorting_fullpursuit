import numpy as np
import os
import platform as sys_platform
import re
import math
from so_sorting.src.sort import reorder_labels
from so_sorting.src import segment
from so_sorting.src.parallel import segment_parallel
from so_sorting.src.overlap import reassign_simultaneous_spiking_clusters, get_zero_phase_kernel, remove_overlapping_spikes
from scipy.signal import fftconvolve




def binary_pursuit(probe_dict, channel, neighbors, neighbor_voltage,
                   event_indices, neuron_labels, clip_width,
                   kernels_path=None, max_gpu_memory=None):
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
    done in np.float32 precision. Output clips are cast to probe_dict['v_dtype']
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

    chan_win, clip_width = segment_parallel.time_window_to_samples(clip_width, probe_dict['sampling_rate'])
    _, master_channel_index, clip_samples, template_samples_per_chan, curr_chan_inds = segment.get_windows_and_indices(
        clip_width, probe_dict['sampling_rate'], channel, neighbors)
    # Remove any spikes within 1 clip width of each other
    event_order = np.argsort(event_indices)
    event_indices = event_indices[event_order]
    neuron_labels = neuron_labels[event_order]

    # Get new aligned multichannel clips here for computing voltage residuals.  Still not normalized
    clips, valid_inds = segment_parallel.get_multichannel_clips(probe_dict, neighbor_voltage, event_indices, clip_width=clip_width)
    event_indices, neuron_labels = segment_parallel.keep_valid_inds([event_indices, neuron_labels], valid_inds)

    # Ensure our neuron_labels go from 0 to M - 1 (required for kernels)
    # This MUST be done after removing overlaps and reassigning simultaneous
    reorder_labels(neuron_labels)

    templates, temp_labels = segment_parallel.calculate_templates(clips, neuron_labels)

    keep_bool = remove_overlapping_spikes(event_indices, clips, neuron_labels, templates,
                                  temp_labels, clip_samples[1]-clip_samples[0])
    event_indices = event_indices[keep_bool]
    neuron_labels = neuron_labels[keep_bool]
    clips = clips[keep_bool, :]
    templates, temp_labels = segment_parallel.calculate_templates(clips, neuron_labels)

    templates = np.vstack(templates).astype(np.float32)
    # Reshape our templates so that instead of being an MxN array, this
    # becomes a 1x(M*N) vector. The first template should be in the first N points
    templates_vector = templates.reshape(templates.size).astype(np.float32)
    # Index of current channel in the neighborhood
    master_channel_index = np.uint32(master_channel_index)
    n_neighbor_chans = np.uint32(neighbors.size)

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

        # Adjust event_indices so that they refer to the start of the clip to be
        # subtracted.
        clip_init_samples = int(np.abs(chan_win[0]))
        event_indices -= clip_init_samples
        event_indices = np.uint32(event_indices)
        neuron_labels = np.uint32(neuron_labels)

        # Determine the segment size that we need to use to fit into the
        # memory on this GPU device.
        gpu_memory_size = device.get_info(cl.device_info.GLOBAL_MEM_SIZE) # Total memory size in bytes
        # Save 50% or 1GB, whichever is less for the operating system to use
        # assuming this is not a headless system.
        gpu_memory_size = max(gpu_memory_size * 0.5, gpu_memory_size - (1024 * 1024 * 1024))
        if max_gpu_memory is not None:
            gpu_memory_size = min(gpu_memory_size, max_gpu_memory)
        # Windows 10 only allows 80% of the memory on the graphics card to be used,
        # the rest is reserved for WDDMv2. See discussion here:
        # https://www.nvidia.com/en-us/geforce/forums/geforce-graphics-cards/5/269554/windows-10wddm-grabbing-gpu-vram/
        regex_version = re.search('[0-9][0-9][.]', sys_platform.version())
        if (sys_platform.system() == 'Windows') and (float(regex_version.group()) >= 10.):
            gpu_memory_size = min(device.get_info(cl.device_info.GLOBAL_MEM_SIZE) * 0.75,
                                  gpu_memory_size)
        assert gpu_memory_size > 0

        # Estimate the number of bytes that 1 second of data take up
        # This is skipping the bias and template squared error buffers which are small
        constant_memory_usage = templates_vector.nbytes + event_indices.nbytes + neuron_labels.nbytes
        # Usage for voltage, additional spike indices, additional spike labels
        memory_usage_per_second = (n_neighbor_chans * probe_dict['sampling_rate'] * np.dtype(np.float32).itemsize +
                                   probe_dict['sampling_rate'] * (np.dtype(np.uint32).itemsize +
                                   np.dtype(np.uint32).itemsize) / template_samples_per_chan)
        # Need to further normalize this by the number of templates and channels
        # More templates and channels slows the GPU algorithm and GPU can timeout
        num_seconds_per_chunk = (gpu_memory_size - constant_memory_usage) \
                                / (templates.shape[0] * n_neighbor_chans * memory_usage_per_second)
        # Do not exceed a the length of a buffer in bytes that can be addressed
        # Note: there is also a max allocation size,
        # device.get_info(cl.device_info.MAX_MEM_ALLOC_SIZE)
        max_addressable_seconds = (((1 << device.get_info(cl.device_info.ADDRESS_BITS)) - 1)
                                    / (np.dtype(np.float32).itemsize * probe_dict['sampling_rate']
                                     * n_neighbor_chans))
        num_seconds_per_chunk = min(num_seconds_per_chunk, np.floor(max_addressable_seconds))
        num_indices_per_chunk = int(np.floor(num_seconds_per_chunk * probe_dict['sampling_rate']))

        if num_indices_per_chunk < 4 * template_samples_per_chan:
            raise ValueError("Cannot fit enough data on GPU to run binary pursuit. Decrease neighborhoods and/or clip width.")

        # Get all of our kernels
        # This kernel subtracts the templates at each of their spike locations
        compute_residual_kernel = prg.compute_residual
        binary_pursuit_kernel = prg.binary_pursuit
        get_adjusted_clips_kernel = prg.get_adjusted_clips

        # Conservatively, do not enqueue more than work group size times the
        # compute units divided by the number of different neuron labels
        # or else systems can timeout and freeze or restart GPU
        resid_local_work_size = compute_residual_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
        pursuit_local_work_size = binary_pursuit_kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)
        max_enqueue_resid = np.uint32(resid_local_work_size * (device.max_compute_units))
        max_enqueue_pursuit = np.uint32(pursuit_local_work_size * (device.max_compute_units))
        max_enqueue_resid = max(resid_local_work_size, resid_local_work_size * (max_enqueue_resid // resid_local_work_size))
        max_enqueue_pursuit = max(pursuit_local_work_size, pursuit_local_work_size * (max_enqueue_pursuit // pursuit_local_work_size))

        # Set up our final outputs
        num_additional_spikes = np.zeros(1, dtype=np.uint32)

        # Set-up any buffers/lists that are not dependent on chunks
        template_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=templates_vector)
        fft_kernels = []
        secret_spike_indices = []
        secret_spike_labels = []
        secret_spike_bool = []
        adjusted_spike_clips = []

        # Compute our template sum squared error (see note below).
        # This is a num_templates x num_neighbors vector
        template_sum_squared = np.zeros(templates.shape[0], dtype=np.float32)
        for n in range(0, templates.shape[0]):
            template_sum_squared[n] = np.float32(-0.5 * np.dot(templates[n, :], templates[n, :]))
            for chan in range(0, n_neighbor_chans):
                t_win = [chan*template_samples_per_chan, chan*template_samples_per_chan + template_samples_per_chan]
                # Also get the FFT kernels used for spike bias below while we're at it
                fft_kernels.append(get_zero_phase_kernel(templates[n, t_win[0]:t_win[1]], clip_init_samples))
        template_sum_squared_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=template_sum_squared)

        # Determine our chunk onset indices, making sure that each new start
        # index overlaps the previous chunk by 3 template widths so that no
        # spikes are missed by binary pursuit
        # Note: Any spikes in the last template width of data are not checked
        chunk_onsets = []
        curr_onset = 0
        while ((curr_onset) < (probe_dict['n_samples'] - template_samples_per_chan)):
            chunk_onsets.append(curr_onset)
            curr_onset += num_indices_per_chunk - 3 * template_samples_per_chan
        print("Using ", len(chunk_onsets), " chunks for binary pursuit.", flush=True)

        # Loop over chunks
        for chunk_number, start_index in enumerate(chunk_onsets):
            stop_index = np.uint32(min(probe_dict['n_samples'], start_index + num_indices_per_chunk))
            print("Starting chunk number", chunk_number, "from", start_index, "to", stop_index, flush=True)
            chunk_voltage = np.float32(neighbor_voltage[:, start_index:stop_index])
            # Reshape voltage over channels into a single 1D vector
            chunk_voltage = chunk_voltage.reshape(chunk_voltage.size)
            select = np.logical_and(event_indices >= start_index, event_indices < stop_index)
            chunk_crossings = event_indices[select] - start_index
            chunk_labels = neuron_labels[select]

            # - Set up the compute_residual kernel -
            # Create our buffers on the graphics cards.
            # Essentially all we are doing is copying each of our arrays to the graphics card.
            voltage_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=chunk_voltage)

            # Set arguments that are the same whether we subtract spikes to
            # get the residual or whether there are no spikes. These will all
            # be used later in the event new spikes are added.
            compute_residual_kernel.set_arg(0, voltage_buffer) # location where chunk voltage is stored
            compute_residual_kernel.set_arg(1, np.uint32(stop_index - start_index)) # length of chunk voltage array
            compute_residual_kernel.set_arg(2, n_neighbor_chans) # Number of neighboring channels
            compute_residual_kernel.set_arg(3, template_buffer) # location where our templates are stored
            compute_residual_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of neurons (`rows` in templates)
            compute_residual_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each channel of templates

            # Only get residual if there are spikes, else it's same as voltage
            if chunk_crossings.shape[0] > 0:
                crossings_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_crossings)
                spike_labels_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chunk_labels)

                # These args are specific to this calculation of residual
                compute_residual_kernel.set_arg(6, crossings_buffer) # Buffer of our spike indices
                compute_residual_kernel.set_arg(7, spike_labels_buffer) # Buffer of our 0 indexed spike labels
                compute_residual_kernel.set_arg(8, np.uint32(chunk_crossings.shape[0])) # Number of crossings

                num_kernels = chunk_crossings.shape[0]
                # Enqueue kernels in groups of max_compute_units * local_work_size
                # Until we exceed num_kernels. On non-interactive systems, we could just
                # enqueue `local_work_size * Integer(ceil(num_kernels / local_work_size))`
                # but this can fail the watch-dog timeout on some Windows systems, forcing
                # us to schedule in smaller units to let the operating system use the
                # card
                total_work_size_resid = np.uint32(resid_local_work_size * np.ceil(num_kernels / resid_local_work_size))
                residual_events = []
                n_to_enqueue = min(total_work_size_resid, max_enqueue_resid)
                next_wait_event = None
                for enqueue_step in np.arange(0, total_work_size_resid, max_enqueue_resid, dtype=np.uint32):
                    residual_event = cl.enqueue_nd_range_kernel(queue,
                                           compute_residual_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    next_wait_event = [residual_event]

                # Set up the binary pursuit OpenCL kernel
                # We can process the voltage trace in small blocks (equal to the template width) by minimizing the RMSE
                # at each interval. For a given index, i, we can add a spike for neuron, n,
                # if:
                #  1) addition of the spike reduces the log-likelihood, which is based
                #     on the Pillow formula:
                #       0.5 * sum((R).^2) - 0.5 * sum((R - template).^2)
                #     where 'R' is the voltage residuals (all other spikes removed).
                #       0.5 * sum(R.^2) - 0.5 * sum(R.^2) + 1 * sum(R .* template) - 0.5 * sum(template.^2)
                #     which is equavalent to
                #       sum(R .* template) - 0.5 * sum(template.^2)
                #     where the last two terms can be computed without a dependence on index i.
                #     NOTE: This formation assumes that the voltage has been scaled by 1/sqrt(sigma) where sigma
                #     is the variance of the residual voltage with the majority of spikes removed. An alternative
                #     definition (which is used here when the voltage has not been scaled) is:
                #       sum(R .* template) - 0.5 * sum(template.^2)
                #  2) the addition of the spike is the largest for the current neuron at
                #     the current timepoint
                #  3) this is the best addition within the template width for any other spike.
                #     That is, we could not add add a spike at an index earlier (by the spike
                #     template width) or later (by the spike template width) that would be
                #     a better addition than the current spike.

                residual_voltage = np.empty(chunk_voltage.shape[0], dtype=np.float32)
                cl.enqueue_copy(queue, residual_voltage, voltage_buffer, wait_for=residual_events) #dest, source
            else:
                # Need this stuff later no matter what
                next_wait_event = None
                residual_voltage = chunk_voltage
                # residual_voltage = np.empty(chunk_voltage.shape[0], dtype=np.float32)
                # cl.enqueue_copy(queue, residual_voltage, voltage_buffer, wait_for=None)

            # Compute the template_sum_squared and bias terms
            spike_biases = np.zeros(templates.shape[0], dtype=np.float32)
            # Compute bias separately for each neuron
            for n in range(0, templates.shape[0]):
                neighbor_bias = np.zeros((stop_index - start_index), dtype=np.float32)
                for chan in range(0, n_neighbor_chans):
                    cv_win = [chan * (stop_index - start_index),
                              chan * (stop_index - start_index) + (stop_index - start_index)]
                    neighbor_bias += np.float32(fftconvolve(
                                residual_voltage[cv_win[0]:cv_win[1]],
                                fft_kernels[n*n_neighbor_chans + chan],
                                mode='same'))
                std_noise = np.median(np.abs(neighbor_bias)) / 0.6745
                spike_biases[n] = 2*std_noise

            # Delete stuff no longer needed for this chunk
            del residual_voltage
            del neighbor_bias
            if chunk_crossings.shape[0] > 0:
                crossings_buffer.release()
                spike_labels_buffer.release()

            num_kernels = np.ceil(chunk_voltage.shape[0] / templates.shape[1])
            total_work_size_pursuit = pursuit_local_work_size * int(np.ceil(num_kernels / pursuit_local_work_size))

            # Construct our buffers
            num_additional_spikes_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=np.uint32)) # NOTE: Must be :rw for atomic to work
            additional_spike_indices_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, size=4 * total_work_size_pursuit) # TODO: How long should this be ?, NOTE: 4 is sizeof('uint32')
            additional_spike_labels_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, size=4 * total_work_size_pursuit) # TODO: How long should this be ?
            spike_biases_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=spike_biases)

            # Construct a local buffer (unsigned int * local_work_size)
            local_buffer = cl.LocalMemory(4 * pursuit_local_work_size)

            binary_pursuit_kernel.set_arg(0, voltage_buffer) # Voltage buffer created previously
            binary_pursuit_kernel.set_arg(1, np.uint32(stop_index - start_index)) # Length of chunk voltage
            binary_pursuit_kernel.set_arg(2, n_neighbor_chans) # number of neighboring channels
            binary_pursuit_kernel.set_arg(3, template_buffer) # Our template buffer, created previously
            binary_pursuit_kernel.set_arg(4, np.uint32(templates.shape[0])) # Number of unique neurons, M
            binary_pursuit_kernel.set_arg(5, np.uint32(template_samples_per_chan)) # Number of timepoints in each template
            binary_pursuit_kernel.set_arg(6, template_sum_squared_buffer) # Sum of squared templated
            binary_pursuit_kernel.set_arg(7, spike_biases_buffer) # Bias
            binary_pursuit_kernel.set_arg(8, local_buffer) # Local buffer
            binary_pursuit_kernel.set_arg(9, num_additional_spikes_buffer) # Output, total number of additional spikes
            binary_pursuit_kernel.set_arg(10, additional_spike_indices_buffer) # Additional spike indices
            binary_pursuit_kernel.set_arg(11, additional_spike_labels_buffer) # Additional spike labels

            # Run the kernel until num_additional_spikes is zero
            chunk_total_additional_spikes = 0
            chunk_total_additional_spike_indices = []
            chunk_total_additional_spike_labels = []
            while True:
                pursuit_events = []
                n_to_enqueue = min(total_work_size_pursuit, max_enqueue_pursuit)
                for enqueue_step in np.arange(0, total_work_size_pursuit, max_enqueue_pursuit, dtype=np.uint32):
                    pursuit_event = cl.enqueue_nd_range_kernel(queue,
                                          binary_pursuit_kernel,
                                          (n_to_enqueue, ), (pursuit_local_work_size, ),
                                          global_work_offset=(enqueue_step, ),
                                          wait_for=next_wait_event)
                    next_wait_event = [pursuit_event]

                cl.enqueue_copy(queue, num_additional_spikes, num_additional_spikes_buffer, wait_for=pursuit_events)
                # print("Added", num_additional_spikes[0], "secret spikes", flush=True)

                if (num_additional_spikes[0] == 0):
                    break # Converged, no spikes added in last pass

                # Read out and save all the new spikes we just found for this chunk
                additional_spike_indices = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                additional_spike_labels = np.zeros(num_additional_spikes[0], dtype=np.uint32)
                cl.enqueue_copy(queue, additional_spike_indices, additional_spike_indices_buffer, wait_for=None)
                cl.enqueue_copy(queue, additional_spike_labels, additional_spike_labels_buffer, wait_for=None)
                chunk_total_additional_spike_indices.append(additional_spike_indices)
                chunk_total_additional_spike_labels.append(additional_spike_labels)

                # We have added additional spikes this pass, so we need to subtract them off
                # by calling the compute_residuals kernel. Most of the arguments are the
                # same as when we called it previously (e.g., voltage is the same buffer on the
                # GPU).
                compute_residual_kernel.set_arg(6, additional_spike_indices_buffer)
                compute_residual_kernel.set_arg(7, additional_spike_labels_buffer)
                compute_residual_kernel.set_arg(8, num_additional_spikes[0])

                total_work_size_resid = resid_local_work_size * int(np.ceil(
                                            (num_additional_spikes[0]) / resid_local_work_size))
                residual_events = []
                n_to_enqueue = min(total_work_size_resid, max_enqueue_resid)
                for enqueue_step in np.arange(0, total_work_size_resid, max_enqueue_resid, dtype=np.uint32):
                    residual_event = cl.enqueue_nd_range_kernel(queue,
                                           compute_residual_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    next_wait_event = [residual_event]
                # Ensure that num_additional_spikes is equal to zero for the next pass
                cl.enqueue_copy(queue, num_additional_spikes_buffer, np.zeros(1, dtype=np.uint32), wait_for=None)
                chunk_total_additional_spikes += num_additional_spikes[0]
                num_additional_spikes[0] = 0

            additional_spike_indices_buffer.release()
            additional_spike_labels_buffer.release()
            spike_biases_buffer.release()

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

                print("Found", chunk_total_additional_spikes, "secret spikes this chunk. Getting adjusted clips for", all_chunk_crossings.shape[0], "total spikes this chunk", flush=True)

                all_adjusted_clips = np.zeros((chunk_total_additional_spikes + chunk_crossings.shape[0]) * templates.shape[1], dtype=np.float32)
                all_chunk_crossings_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_crossings)
                all_chunk_labels_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=all_chunk_labels)
                all_adjusted_clips_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=all_adjusted_clips)

                get_adjusted_clips_kernel.set_arg(0, voltage_buffer)
                get_adjusted_clips_kernel.set_arg(1, np.uint32(stop_index - start_index))
                get_adjusted_clips_kernel.set_arg(2, n_neighbor_chans)
                get_adjusted_clips_kernel.set_arg(3, template_buffer)
                get_adjusted_clips_kernel.set_arg(4, np.uint32(templates.shape[0]))
                get_adjusted_clips_kernel.set_arg(5, np.uint32(template_samples_per_chan))
                get_adjusted_clips_kernel.set_arg(6, all_chunk_crossings_buffer)
                get_adjusted_clips_kernel.set_arg(7, all_chunk_labels_buffer)
                get_adjusted_clips_kernel.set_arg(8, np.uint32(all_chunk_crossings.shape[0]))
                get_adjusted_clips_kernel.set_arg(9, all_adjusted_clips_buffer)

                # Run the kernel using same sizes as for resid kernel
                total_work_size_clips = resid_local_work_size * int(np.ceil(
                                            (all_chunk_crossings.shape[0]) / resid_local_work_size))
                clip_events = []
                n_to_enqueue = min(total_work_size_clips, max_enqueue_resid)
                print("Getting adjusted clips", flush=True)
                for enqueue_step in np.arange(0, total_work_size_clips, max_enqueue_resid, dtype=np.uint32):
                    clip_event = cl.enqueue_nd_range_kernel(queue,
                                           get_adjusted_clips_kernel,
                                           (n_to_enqueue, ), (resid_local_work_size, ),
                                           global_work_offset=(enqueue_step, ),
                                           wait_for=next_wait_event)
                    next_wait_event = [clip_event]
                cl.enqueue_copy(queue, all_adjusted_clips, all_adjusted_clips_buffer, wait_for=clip_events)
                all_adjusted_clips = np.reshape(all_adjusted_clips, (all_chunk_crossings.shape[0], templates.shape[1]))

                # Save all chunk data and free spike dependent memory
                secret_spike_indices.append(all_chunk_crossings + start_index)
                secret_spike_labels.append(all_chunk_labels)
                adjusted_spike_clips.append(all_adjusted_clips)
                all_chunk_bool = np.zeros(all_chunk_crossings.shape[0], dtype=np.bool)
                all_chunk_bool[0:chunk_total_additional_spikes] = True
                secret_spike_bool.append(all_chunk_bool)

                all_chunk_crossings_buffer.release()
                all_chunk_labels_buffer.release()
                all_adjusted_clips_buffer.release()

            # Release buffers before going on to next chunk
            voltage_buffer.release()
            num_additional_spikes_buffer.release()

        template_buffer.release()
        template_sum_squared_buffer.release()

    event_indices = np.int64(np.hstack(secret_spike_indices))
    neuron_labels = np.int64(np.hstack(secret_spike_labels))
    adjusted_clips = (np.vstack(adjusted_spike_clips)).astype(probe_dict['v_dtype'])
    binary_pursuit_spike_bool = np.hstack(secret_spike_bool)
    # Realign events with center of spike
    event_indices += clip_init_samples

    print("Found a total of", np.count_nonzero(binary_pursuit_spike_bool), "secret spikes", flush=True)

    return event_indices, neuron_labels, binary_pursuit_spike_bool, adjusted_clips
