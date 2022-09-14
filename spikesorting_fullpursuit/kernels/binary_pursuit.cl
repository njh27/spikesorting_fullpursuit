/**
 * @file
 * @author David J. Herzfeld <herzfeldd@gmail.com>
 *
 * @section LICENSE
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details at <http://www.gnu.org/licenses/>
 *
 * @section DESCRIPTION
 */

/* Allow customization of the memory type for the GPU */
#ifndef voltage_type
#define voltage_type float
#endif

/* Define NULL if it is not already defined by the compiler */
#ifndef NULL
#define NULL ((void *)0)
#endif

/*----------------------------------------------------------------------------
 * NOTES ON PACKING OF VOLTAGES (E.G., `VOLTAGE` and `TEMPLATES` vectors):
 *
 * All code in this file assumes the same packaging of voltage related vectors.
 * Namely, the input parameter `voltage_length` refers to the length of the
 * voltage vector for a SINGLE CHANNEL. The total length of the binary voltage
 * should be voltage_length * num_neighbor_channels (neighbors are packed into
 * the same voltage vector one after the other).
 *
 * Similarly, the binary representation of templates for these functions is
 * a contiguous floating point array with a total binary length of
 * num_timepoints * num_channels * num_neurons.
 * The packing looks like:
 *   [(num_timepoints of neuron 1 on channel 1)...
 *    ...(num_timepoints of neuron 1 on channel 2)...
 *    ...(num_timepoints of neuron 1 on channel 3)...
 *    ...(num_timepoints of neuron 2 on channel 1)...
 *    ...(num_timepoints of neuron 2 on channel 2)...etc..]
 *---------------------------------------------------------------------------*/

/**
 * Kernel used to subtract existing spikes from the voltage waveform, leaving
 * only the residual voltage.
 *
 * IMPORTANT NOTE: This function assumes that any spike index is more than
 * template_length indices away from its nearest neighbors. That is, if there is
 * a spike at index 100, there can be no other spikes in the list from 100 -
 * template_length to 100 + length_length. This is a HARD REQUIREMENT. If this
 * requirement is not met, the answer is not guaranteed to be correct due to a
 * race condition during the subtraction (noted in the function). Importantly,
 * consecurive calls of the binary_pursuit kernel and this kernel will function
 * correctly as spike indices found during the binary_pursuit kernel are gauranteed
 * to be template_length apart on each pass.
 *
 * Parameters:
 *  voltage: A float32 vector containing the voltage data
 *  voltage_length: The length of voltage for a single channel
 *  num_neighbor_channels: The number of neighbors channels packed into voltage and templates
 *  templates: A num_timepoints * num_channels * num_neurons vector (see above description)
 *  num_templates: The dimension M of templates (num_neurons)
 *  template_length: The dimension N of templates (num timepoints)
 *  spike_indices: A vector containing the initial index of each spike to remove. Each
 *   index is the FIRST value where we should subtract the first value of the template.
 *  spike_label: The label (starting at 0 NOT 1) for each spike as an index into templates.
 *  spike_indices_length: The number of spike indices (and length of spike labels).
 */
__kernel void compute_residual(
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    __global const unsigned int * spike_indices,
    __global const unsigned int * spike_labels,
    const unsigned int spike_indices_length)
{
    const size_t id = get_global_id(0);
    /* Return if we started a kernel and it has nothing to do */
    if (id >= spike_indices_length || num_neighbor_channels == 0)
    {
        return; /* Do nothing */
    }

    unsigned int i, current_channel;
    const unsigned int spike_index = spike_indices[id]; /* The first index of our spike */
    const unsigned int spike_label = spike_labels[id]; /* Our template label (row in templates) */
    if (spike_label >= num_templates)
    {
        return; /* this is technically an error. This label does not exist in templates */
    }

    for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
    {
        unsigned int voltage_offset = spike_index + (voltage_length * current_channel); /* offset in voltage for this channel channel */
        unsigned int template_offset = (spike_label * template_length * num_neighbor_channels) + (current_channel * template_length);
        for (i = 0; i < template_length; i++)
        {
            /* NOTE: There is technically a race condition here if two items */
            /* in spike_indices contain indices that are within template_length of each other */
            /* The answer will not be gauranteed to be correct since we are not performing an atomic */
            /* subtraction. See note at the top of this function */
            if (spike_index + i >= voltage_length)
            {
                break;
            }
            voltage[voltage_offset + i] -= templates[template_offset + i];
        }
    }

    return;
}


/**
 * Helper function that computes the prefix sum in a local buffer (in place)
 *
 * The prefix sum is the cummulative sum of all previous elements.
 * By definition the prefix sum at index zero must be zero.
 *
 * Imagine that the input, x, is the following vector:
 *  x = [1, 3, 0, 2, 4]
 * The in-place result after calling this function is:
 *  x = [0, 1, 4, 4, 6]
 * This function is useful when you want a worker to compute it's index
 * into a local array such that the output results are ordered appropriately.
 */
static void prefix_local_sum(__local unsigned int * restrict x) /**< Length must be equal to local size */
{
    const size_t local_index = get_local_id(0);
    const size_t local_size = get_local_size(0);

    /* Reduction */
    unsigned int stride = 1;
    unsigned int max_size = local_size >> 1; /* Divide by 2 */
    while (stride <= max_size)
    {
        unsigned int index = (local_index + 1) * stride * 2 - 1;
        if (index < local_size)
        {
            x[index] += x[index - stride];
        }
        stride = stride << 1; /* Multiply by two */
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    /* Second pass */
    if (local_index == 0)
    {
        x[local_size - 1] = 0; /* Zero at end */
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    stride = local_size >> 1; /* Divide by 2 */
    while (stride > 0)
    {
        unsigned int index = (local_index + 1) * stride * 2 - 1;
        if (index < local_size)
        {
            unsigned int temp = x[index];
            x[index] += x[index - stride];
            x[index-stride] = temp;
        }
        stride = stride >> 1; /* Divide by 2 */
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    return;
}


/**
 * Peforms a reduction on the local float array x
 * The maximum value of x is stored in x[0]. The corresponding element of x_id
 * is stored in x_id[0]
 * The size of x and x_id should be equal to the local item size
*/
static void max_local_reduction(__local float *x, __local unsigned int *x_id)
{
    const unsigned int local_size = get_local_size(0);
    const unsigned int local_id = get_local_id(0);

    unsigned int num_active = local_size >> 1; /* Divide by two */
    while (num_active > 0)
    {
        if (local_id < num_active)
        {
            if (x[local_id + num_active] > x[local_id])
            {
                x[local_id] = x[local_id + num_active];
                x_id[local_id] = x_id[local_id + num_active];
            }
        }
        num_active = num_active >> 1;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


/**
 * A helper function which computes the delta likelihood given
 * the residual voltages. The output of this function is
 * the the maximum likelihood for adding any of the templates at
 * the current index as well as the neuron that determined that
 * maximum likelihood.
 *
 * Note: If the length of the templates exceeds the length of the
 * voltage value given the current index, this function returns (0, 0).
 *
 * The function first looks only over the master channel. If, and only if,
 * the maximum likelihood on the current channel exceeds zero do we check
 * across the remaining channels. The returned maximum likelihood is
 * the likelihood across all channels.
 *
 * Parameters:
 *  voltage: The global voltage
 *  voltage_length: The length of the local voltage in samples
 *  num_neighbor_channels: The number of neighbors channels packed into voltage and templates
 *  index: The index within voltage to start our maximum likelihood computation.
 *  templates: A num_timepoints * num_channels * num_neurons vector (see above description)
 *  num_templates: The dimension M of templates (num_neurons)
 *  template_length: The dimension N of templates (num timepoints)
 *  template_sum_squared: A 1xnum_templates vector containing the template sum squared values
 *  gamma: 1xnum_templates vector containing the bias parameter
 *  maximum_likelihood [out]: The maximum likelihood found at this point (private).
 */
static float compute_maximum_likelihood(
    __global voltage_type * restrict voltage,
    unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    const unsigned int index,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    const unsigned int template_number,
    __global const float * restrict template_sum_squared)
{
    unsigned int i;
    unsigned int current_channel;
    /* Set our maximum likelihood found to 0.0 (no spike to add) */
    float maximum_likelihood = 0.0;
    if (index + template_length >= voltage_length)
    {
        return maximum_likelihood; /* Returns maximum likelihood = 0.0 */
    }
    if (template_number >= num_templates)
    {
        return maximum_likelihood; /* Invalid template number, return 0.0 */
    }
    maximum_likelihood = template_sum_squared[template_number];

    /* The master channel exceeded threshold, check all of our neighbors */
    for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
    {
        unsigned int template_offset = (template_number * template_length * num_neighbor_channels) + (current_channel * template_length);
        unsigned int voltage_offset = index + (voltage_length * current_channel);
        for (i = 0; i < template_length; i++)
        {
            maximum_likelihood = maximum_likelihood + (float) templates[template_offset + i] * (float) voltage[voltage_offset + i];
        }
    }
    return maximum_likelihood;
}


__kernel void compute_full_likelihood(
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    __global const float * restrict template_sum_squared,
    __global const unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global float * restrict full_likelihood_function)
{
    if (num_neighbor_channels == 0)
    {
        return; /* Invalid number of channels (must be >= 1) */
    }
    const size_t global_id = get_global_id(0);
    __private const size_t window_id = (size_t) (global_id / ((size_t) num_templates));
    if (num_window_indices > 0 && window_indices != NULL && window_id >= num_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    __private const unsigned int id = window_indices[window_id];
    __private const unsigned int template_number = (unsigned int) (global_id % ((size_t) num_templates));
    if (template_number >= num_templates)
    {
        return; /* Invalid template number */
    }

    /* Only spikes found within [id * template_length, id + 1 * template_length] are added to the output */
    /* All other spikes are ignored (previous window and next window) */
    __private const unsigned int start_of_my_window = id * template_length;
    __private const unsigned int end_of_my_window = ((id + 1) * template_length) > voltage_length ? voltage_length : ((id + 1) * template_length);
    __private const unsigned int full_likelihood_function_offset = template_number * voltage_length + start_of_my_window;

    unsigned int i;
    /* Compute our minimum cost to go for adding our current template at each point in the window */
    for (i = 0; i < (unsigned) end_of_my_window - start_of_my_window; i++)
    {
        float current_maximum_likelihood = compute_maximum_likelihood(voltage, voltage_length, num_neighbor_channels,
            i + start_of_my_window, templates, num_templates, template_length, template_number,
            template_sum_squared);
        full_likelihood_function[full_likelihood_function_offset + i] = current_maximum_likelihood;
    }
    return;
}


/**
 * Kernel used to identify the template and index that maximimizes the
 * likelihood within a small window. This should be called successively,
 * with an increasing template_number (0...num_templates-1). All other parameters
 * should remain the same between calls.
 *
 * Each kernel is assigned a window whose length is equal to template_length. The kernel
 * checks the preceeding window and the next window to determine if a spike should
 * be added in any of these three windows. If a spike should be added in the current
 * kernel's window, the spike is added to the output and subtracted from voltage. However,
 * if either the preceeding window or the next window show a larger maximum likelihood,
 * we return without doing anytime (believing that a seperate kerel will take care
 * of those spikes).
 *
 * On the first pass through binary pursuit, this function should be called for
 * all non-overlapping windows in the voltage trace. This should be done by
 * passing in 0:num_windows - 1 as the window_indices (or NULL). On subsequent calls,
 * window_indices can be reduced to only the windows that need to be checked again (e.g.,
 * those windows that had a positive likelihood on the previous pass). To help reduce
 * the computation, the check_window_on_next_pass vector stores a boolean array
 * that can be used to compute the window indices for the next pass through the
 * complete binary pursuit algorithm.
 *
 * Parameters:
 *  voltage: A voltage_type (typically float32) vector containing the voltage data
 *  voltage_length: The length of voltage
 *  num_neighbor_channels: The number of neighbors channels packed into voltage and templates
 *  templates: A num_timepoints * num_channels * num_neurons vector (see above description)
 *  num_templates: The dimension M of templates (num_neurons)
 *  template_length: The dimension N of templates (num timepoints)
 *  template_sum_squared: A 1xnum_templates vector containing the template sum squared values
 *  template_number: The index in the templates vector to compute the max likelihood for
 *  window_indices: A list of window numbers (0, 1, 2, 3,... each referring to a non-overlapping
 *   window of width template_length. If this is null, we use the global id.
 *  num_window_indices: The length of the window indices vector. If this value is zero, we
 *   use the global id as our window index.
 *  gamma: 1xnum_templates vector containing the bias parameter
 *  best_spike_indices [out]: A global vector (max length = 1xnum_workers) containing the indices within voltage
 *   within each window that have the best likelihood to add.
 *  best_spike_labels [out]: The template ids that we have added at additional spike_indices. Same
 *   restrictions as additional_spike_indices.
 *  check_window_on_next_pass [out]: A boolean array of type UInt8 that can be used
 *   to reduce the number of workers that need to be run on each pass. The indices
 *   where this vector is > 0, correspond to the work ids that need to be run in the
 *   next pass (e.g., if we "find(check_window_on_next_pass > 0)", we can pass this
 *   list of indices into the next round, reducing the computation.
 */
__kernel void compute_template_maximum_likelihood(
    const unsigned int voltage_length,
    const unsigned int template_length,
    const unsigned int template_number,
    __global const unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global float * restrict best_spike_likelihoods,
    __global unsigned char * restrict check_window_on_next_pass,
    __global unsigned char * restrict overlap_recheck,
    __global unsigned int * restrict overlap_best_spike_indices,
    __global unsigned int * restrict overlap_best_spike_labels,
    __global float * restrict full_likelihood_function,
    __global const float * restrict likelihood_lower_thresholds)
{
    const size_t global_id = get_global_id(0);
    if (num_window_indices > 0 && window_indices != NULL && global_id >= num_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    __private const unsigned int id = (num_window_indices > 0 && window_indices != NULL) ? window_indices[global_id] : global_id;

    /* Only spikes found within [id * template_length, id + 1 * template_length] are added to the output */
    /* All other spikes are ignored (previous window and next window) */
    __private const unsigned int start_of_my_window = id * template_length;
    __private const unsigned int end_of_my_window = (id + 1) * template_length;
    __private const unsigned int start = (template_length > start_of_my_window) ? 0 : (start_of_my_window - template_length);
    __private const unsigned int stop = (end_of_my_window + template_length) > voltage_length ? voltage_length : (end_of_my_window + template_length);
    __private const unsigned int full_likelihood_function_offset = template_number * voltage_length + start;
    if (end_of_my_window >= voltage_length - template_length)
    {
        return; /* Nothing to do, the end of the current window exceeds the bounds of voltage */
    }

    /* Cache the data from the global buffers so that we only have to write to them once */
    __private float best_spike_likelihood_private = best_spike_likelihoods[id];
    __private unsigned int best_spike_label_private = best_spike_labels[id];
    __private unsigned int best_spike_index_private = best_spike_indices[id];
    if (template_number == 0)
    {
        /* Reset our best_spike_likelihoods*/
        best_spike_likelihood_private = 0.0;
    }

    unsigned int i;
    float current_maximum_likelihood = 0.0;
    unsigned char check_window = 0;
    /* Compute our minimum cost to go for adding our current template at each point in the window */
    for (i = 0; i < (unsigned) stop - start; i++)
    {
        current_maximum_likelihood = full_likelihood_function[full_likelihood_function_offset + i];
        if ( (current_maximum_likelihood > best_spike_likelihood_private) )
            // && (current_maximum_likelihood > likelihood_lower_thresholds[template_number]) )
        {
            best_spike_likelihood_private = current_maximum_likelihood;
            best_spike_label_private = template_number;
            best_spike_index_private = start + i;
        }
        if ( (current_maximum_likelihood > 0.0)
        // if ( (1)
        // if ( (current_maximum_likelihood > likelihood_lower_thresholds[template_number])
            && (start + i >= start_of_my_window) && (start + i < end_of_my_window) )
        {
            /* Track windows that need checked next pass regardless of whether */
            /* they end up having a spike added */
            check_window = 1;
        }
    }
    if (check_window && check_window_on_next_pass != NULL)
    {
        check_window_on_next_pass[id] = 1;
    }
    overlap_recheck[id] = 0;
    /* Must set overlap_recheck to 1 ONLY IF a spike will be added in this */
    /* window by binary pursuit. */
    // if ((best_spike_likelihood_private > likelihood_lower_thresholds[best_spike_label_private]) && (best_spike_index_private >= start_of_my_window) && (best_spike_index_private < end_of_my_window))
    // {
    //     overlap_recheck[id] = 1;
    // }
    if ( (best_spike_likelihood_private > 0.0) && (best_spike_index_private >= start_of_my_window) && (best_spike_index_private < end_of_my_window))
    {
        overlap_recheck[id] = 1;
    }
    /* Write our results back to the global vectors */
    best_spike_likelihoods[id] = best_spike_likelihood_private;
    best_spike_labels[id] = best_spike_label_private;
    best_spike_indices[id] = best_spike_index_private;
    overlap_best_spike_indices[id] = best_spike_index_private;
    overlap_best_spike_labels[id] = best_spike_label_private;
    return;
}


__kernel void overlap_recheck_indices(
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    __global const unsigned int * restrict overlap_window_indices,
    const unsigned int num_overlap_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global float * restrict full_likelihood_function,
    const unsigned int n_max_shift_inds,
    __local float * restrict local_likelihoods,
    __local unsigned int * restrict local_ids,
    __global float * restrict overlap_group_best_likelihood,
    __global unsigned int * restrict overlap_group_best_work_id)
{
    const size_t local_id = get_local_id(0);
    const size_t local_size = get_local_size(0);
    /* Make sure everyone starts at 0.0 */
    local_likelihoods[local_id] = 0.0;
    local_ids[local_id] = (unsigned int) local_id;

    /* Need to track valid workers. Can't return because everything must hit barrier */
    unsigned char skip_curr_id = 0;

    if (num_neighbor_channels == 0)
    {
        return; /* Invalid number of channels (must be >= 1) */
    }

    const size_t global_id = get_global_id(0);
    /* Can't do this stuff without num_shifts > 0. Add 1 to include shift 0 */
    __private size_t num_shifts = (size_t) (2 * n_max_shift_inds + 1);
    /* Need basic info about crazy indexing for each worker */
    __private size_t items_per_index = num_shifts * num_shifts * (size_t) num_templates;
    /* Round ceiling gives number of work groups needed per recheck index */
    __private size_t n_local_ID = (size_t) (((items_per_index - 1) / local_size)+1);
    /* Number of leftover workers for each index, used as offset */
    __private size_t n_local_ID_leftover = (size_t) (local_size * n_local_ID) % items_per_index;

    /* Get the overall group ID for this group */
    __private const unsigned int group_id = (unsigned int) (global_id / local_size);
    if (group_id > num_overlap_window_indices * n_local_ID)
    {
        return; /* This entire group is an extra group with nothing to do */
    }

    __private size_t id_index = (size_t) (global_id / (n_local_ID * local_size));
    if (id_index >= num_overlap_window_indices)
    {
        skip_curr_id = 1; /* Extra worker with nothing to do (shouldn't happen if input is correct)*/
    }

    /* Need this set up top in case we skip */
    __private float current_maximum_likelihood = 0.0;
    __private unsigned int template_number;
    __private unsigned int fixed_shift_ref_ind, template_shift_ref_ind;
    __private unsigned int best_spike_label_private, best_spike_index_private;
    __private const size_t offset_global_id = global_id - id_index * n_local_ID_leftover;

    /* Avoid invalid indexing */
    if (skip_curr_id == 0)
    {
        /* Cache the data from the global buffers so that we only have to write to them once */
        const size_t id = overlap_window_indices[id_index];
        best_spike_label_private = best_spike_labels[id];
        best_spike_index_private = best_spike_indices[id];
        if (best_spike_index_private >= (voltage_length - template_length))
        {
            skip_curr_id = 1;
        }

        /* Figure out the template and shifts for this worker based on crazy indexing */
        template_number = (unsigned int) (offset_global_id % (size_t) num_templates);
        fixed_shift_ref_ind = (unsigned int) ((offset_global_id / (size_t) num_templates) % num_shifts);
        template_shift_ref_ind = (unsigned int) ((offset_global_id / (num_shifts * (size_t) num_templates)) % num_shifts);
        /* Check our current shifts are in bounds for both shifts */
        if ((fixed_shift_ref_ind + best_spike_index_private < n_max_shift_inds) || (fixed_shift_ref_ind + best_spike_index_private >= (voltage_length - template_length + n_max_shift_inds)))
        {
            skip_curr_id = 1;
        }
        if ((template_shift_ref_ind + best_spike_index_private < n_max_shift_inds) || (template_shift_ref_ind + best_spike_index_private >= (voltage_length - template_length + n_max_shift_inds)))
        {
            skip_curr_id = 1;
        }
        /* Check if shifts are so extreme templates are no longer overlapping */
        if (fixed_shift_ref_ind > template_shift_ref_ind)
        {
            if (fixed_shift_ref_ind - template_shift_ref_ind > template_length)
            {
                skip_curr_id = 1;
            }
        }
        if (template_shift_ref_ind > fixed_shift_ref_ind)
        {
            if (template_shift_ref_ind - fixed_shift_ref_ind > template_length)
            {
                skip_curr_id = 1;
            }
        }
    }

    /* Compiler demands these outside of skip check */
    __private unsigned int absolute_fixed_index = 0;
    __private unsigned int absolute_shift_index = 0;
    if (skip_curr_id == 0)
    {
        absolute_fixed_index = best_spike_index_private + fixed_shift_ref_ind - n_max_shift_inds;
        absolute_shift_index = best_spike_index_private + template_shift_ref_ind - n_max_shift_inds;

        __private const float best_spike_likelihood = full_likelihood_function[best_spike_label_private * voltage_length + absolute_fixed_index];
        __private const float template_spike_likelihood = full_likelihood_function[template_number * voltage_length + absolute_shift_index];

        /* Do not do this if subtracting either unit at its current index does */
        /* not improve the likelihood */
        if ((best_spike_likelihood <= 0) &&
              (template_spike_likelihood <= 0))
        {
            skip_curr_id = 1;
        }
    }
    if (skip_curr_id == 0)
    {
        __private unsigned int shift_template_offset, fixed_template_offset, voltage_offset;
        __private float curr_fixed_value, curr_shift_value, curr_summed_value;
        __private unsigned int j, current_channel, fixed_first_j, shift_first_j;

        /* Compute reference and offset indices for computing the overlap template within original window */
        __private signed int curr_fixed_align = (signed int) best_spike_index_private - (signed int) absolute_fixed_index;
        fixed_first_j = 0;
        if (curr_fixed_align < 0)
        {
            fixed_first_j = -1 * curr_fixed_align;
        }
        __private signed int curr_shift_align = (signed int) best_spike_index_private - (signed int) absolute_shift_index;
        shift_first_j = 0;
        if (curr_shift_align < 0)
        {
            shift_first_j = -1 * curr_shift_align;
        }

        __private float summed_chan_ss;
        for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
        {
            fixed_template_offset = (best_spike_label_private * template_length * num_neighbor_channels) + (current_channel * template_length);
            shift_template_offset = (template_number * template_length * num_neighbor_channels) + (current_channel * template_length);
            voltage_offset = best_spike_index_private + (voltage_length * current_channel);

            summed_chan_ss = 0.0;
            for (j = 0; j < template_length; j++)
            {
                /* Doing all this to avoid branch divergence of if else statements */
                curr_fixed_value = 0.0;
                if ((j >= fixed_first_j) && (curr_fixed_align + j < template_length))
                {
                    curr_fixed_value = templates[fixed_template_offset + curr_fixed_align + j];
                }
                curr_shift_value = 0.0;
                if ((j >= shift_first_j) && (curr_shift_align + j < template_length))
                {
                    curr_shift_value = templates[shift_template_offset + curr_shift_align + j];
                }
                /* Value of summed template on this channel at index j */
                curr_summed_value = curr_fixed_value + curr_shift_value;
                summed_chan_ss += curr_summed_value * curr_summed_value;
                current_maximum_likelihood += curr_summed_value * voltage[voltage_offset + j];
            }

            /* Now that we have full template sum square for this channel */
            /* multiply by this channel's bias and add to bias term */
            if (summed_chan_ss > 0.0)
            {
                current_maximum_likelihood -= 0.5 * summed_chan_ss;
            }
        }
    }

    local_likelihoods[local_id] = current_maximum_likelihood;

    /* Wait for all workers to get here */
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Reduction to find max likelihood and id of best worker in this group */
    max_local_reduction(local_likelihoods, local_ids);
    /* Leave it to first worker to write best results to global buffer */
    if (local_id == 0)
    {
        /* Get the group number relative to current id_index being checked */
        __private const unsigned int curr_index_group = group_id % n_local_ID;
        /* This gives the index of the best work such that we can later */
        /* determine the shifts and templates used for best worker. */
        /* NOTE: This removes any information about the global ID of the best */
        /* worker because it is not needed to compute the shifts and templates. */
        /* The hope is this reduces a possible size_t to the uint32 of the buffer */
        overlap_group_best_likelihood[group_id] = local_likelihoods[0];
        overlap_group_best_work_id[group_id] = local_ids[0] + (unsigned int) local_size * curr_index_group;
    }
    return;
}


__kernel void parse_overlap_recheck_indices(
    const unsigned int voltage_length,
    const unsigned int num_templates,
    __global const unsigned int * restrict overlap_window_indices,
    const unsigned int num_overlap_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global float * restrict best_spike_likelihoods,
    __global unsigned int * restrict overlap_best_spike_indices,
    __global unsigned int * restrict overlap_best_spike_labels,
    __global float * restrict full_likelihood_function,
    const unsigned int n_max_shift_inds,
    __global const float * restrict likelihood_lower_thresholds,
    __global float * restrict overlap_group_best_likelihood,
    __global unsigned int * restrict overlap_group_best_work_id,
    __global unsigned char * restrict overlap_recheck)
{

    __private const size_t num_shifts = (size_t) (2 * n_max_shift_inds + 1);
    const size_t global_id = get_global_id(0);
    if (num_overlap_window_indices > 0 && overlap_window_indices != NULL && global_id >= num_overlap_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    const size_t id = overlap_window_indices[global_id];

    /* Need basic info about crazy indexing for each worker */
    const size_t local_size = get_local_size(0); /* MUST BE THE SAME AS OVERLAP_RECHECK_INDICES KERNEL */
    __private const size_t items_per_index = num_shifts * num_shifts * (size_t) num_templates;
    /* Round ceiling gives number of work groups needed per recheck index */
    __private const size_t n_local_ID = (size_t) (((items_per_index - 1) / local_size)+1);
    /* Number of leftover workers for each index, used as offset */

    __private float best_spike_likelihood_private = best_spike_likelihoods[id];
    __private unsigned int best_spike_label_private = best_spike_labels[id];
    __private unsigned int best_spike_index_private = best_spike_indices[id];
    __private float best_group_likelihood = best_spike_likelihood_private;
    __private unsigned int best_template_shifts_id = 0;

    __private const unsigned int start_id_index = (unsigned int) (n_local_ID * global_id);
    __private unsigned int search_index;
    /* Search over the work group results corresponding to our current id */
    for (search_index=start_id_index; search_index < (start_id_index + n_local_ID); search_index++)
    {
        if (overlap_group_best_likelihood[search_index] > best_group_likelihood)
        {
            best_group_likelihood = overlap_group_best_likelihood[search_index];
            best_template_shifts_id = overlap_group_best_work_id[search_index];
        }
    }
    if ((best_group_likelihood <= best_spike_likelihood_private) || (best_group_likelihood <= 0.0))
    {
        overlap_recheck[id] = 0;
        return; /* Shifts didn't improve so just return the previous values */
    }

    /* Shifts improved likelihood, so we need to use best shifts ID and crazy indices to find answer */
    if ((best_template_shifts_id % (n_local_ID * local_size)) >= items_per_index)
    {
        return; /* Extra worker with nothing to do */
    }

    /* Figure out the template and shifts for this worker based on crazy indexing */
    __private const unsigned int template_number = (unsigned int) (best_template_shifts_id % (size_t) num_templates);
    __private const signed int fixed_shift_ref_ind = (unsigned int) ((best_template_shifts_id / (size_t) num_templates) % num_shifts);
    __private const signed int template_shift_ref_ind = (unsigned int) ((best_template_shifts_id / (num_shifts * (size_t) num_templates)) % num_shifts);

    if ( (best_group_likelihood < likelihood_lower_thresholds[best_spike_label_private])
        || (best_group_likelihood < likelihood_lower_thresholds[template_number]) )
    {
        overlap_recheck[id] = 0;
        return; /* Combined template doesn't exceed both units' thresholds */
    }

    /* These should all be in bounds or overlap_recheck_indices wouldn't have */
    /* likelihood > 0 */
    __private const unsigned int absolute_fixed_index = best_spike_index_private + fixed_shift_ref_ind - n_max_shift_inds;
    __private const unsigned int absolute_shift_index = best_spike_index_private + template_shift_ref_ind - n_max_shift_inds;

    float actual_template_likelihood_at_index = full_likelihood_function[best_spike_label_private * voltage_length + absolute_fixed_index];
    float actual_current_maximum_likelihood = full_likelihood_function[template_number * voltage_length + absolute_shift_index];

    /* Reset the likelihood and best index and label to maximum.
      These need to be reset only if the new index can pass the
      threshold check in "binary_pursuit" below. Otherwise this
      spike won't get added but it will continue to be checked. */
    if ((actual_template_likelihood_at_index >= actual_current_maximum_likelihood)
        && (actual_template_likelihood_at_index > 0.0))
    {
        /* The main label has better likelihood than best shifted match */
        best_spike_likelihoods[id] = best_group_likelihood;
        overlap_best_spike_labels[id] = best_spike_label_private;
        overlap_best_spike_indices[id] = absolute_fixed_index;
    }
    else if ((actual_current_maximum_likelihood > actual_template_likelihood_at_index )
        && (actual_current_maximum_likelihood > 0.0))
    {
        /* The best shifted match unit has better likelihood than the main label */
        best_spike_likelihoods[id] = best_group_likelihood;
        overlap_best_spike_labels[id] = template_number;
        overlap_best_spike_indices[id] = absolute_shift_index;
    }
    // if ((actual_template_likelihood_at_index >= actual_current_maximum_likelihood)
    //     && (actual_template_likelihood_at_index > likelihood_lower_thresholds[best_spike_label_private]))
    // {
    //     /* The main label has better likelihood than best shifted match */
    //     best_spike_likelihoods[id] = actual_template_likelihood_at_index;
    //     overlap_best_spike_labels[id] = best_spike_label_private;
    //     overlap_best_spike_indices[id] = absolute_fixed_index;
    // }
    // else if ((actual_current_maximum_likelihood > actual_template_likelihood_at_index )
    //     && (actual_current_maximum_likelihood > likelihood_lower_thresholds[template_number]))
    // {
    //     /* The best shifted match unit has better likelihood than the main label */
    //     best_spike_likelihoods[id] = actual_current_maximum_likelihood;
    //     overlap_best_spike_labels[id] = template_number;
    //     overlap_best_spike_indices[id] = absolute_shift_index;
    // }
    // else if (actual_template_likelihood_at_index > likelihood_lower_thresholds[best_spike_label_private])
    // {
    //     /* The main label exceeds threshold */
    //     best_spike_likelihoods[id] = actual_template_likelihood_at_index;
    //     overlap_best_spike_labels[id] = best_spike_label_private;
    //     overlap_best_spike_indices[id] = absolute_fixed_index;
    // }
    // else if (actual_current_maximum_likelihood > likelihood_lower_thresholds[template_number])
    // {
    //     /* The best shifted match exceeds threshold */
    //     best_spike_likelihoods[id] = actual_current_maximum_likelihood;
    //     overlap_best_spike_labels[id] = template_number;
    //     overlap_best_spike_indices[id] = absolute_shift_index;
    // }
    else
    {
        /* Says "do nothing" so we stick with our original spike index
        and require that it exceeds threshold by setting overlap_recheck
        to 0. */
        overlap_best_spike_indices[id] = best_spike_indices[id];
        overlap_best_spike_labels[id] = best_spike_labels[id];
        overlap_recheck[id] = 0;
    }
}


/* If the new shifted index is outside
the current window for this worker, it will check whether a spike was added
+/- 2 windows away (the closest a spike could have been added), and if so, will
return, kicking the recheck can down the road until the next pass. */
__kernel void check_overlap_reassignments(
    const unsigned int voltage_length,
    const unsigned int template_length,
    __global const unsigned int * restrict overlap_window_indices,
    const unsigned int num_overlap_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global unsigned char * restrict check_window_on_next_pass,
    __global unsigned int * restrict overlap_best_spike_indices,
    __global unsigned int * restrict overlap_best_spike_labels,
    __global unsigned char * restrict overlap_recheck)
{
    const size_t global_id = get_global_id(0);
    if (num_overlap_window_indices > 0 && overlap_window_indices != NULL && global_id >= num_overlap_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    const size_t id = (num_overlap_window_indices > 0 && overlap_window_indices != NULL) ? overlap_window_indices[global_id] : global_id;
    const unsigned int start_of_my_window = ((signed int) id) * ((signed int) template_length);
    const unsigned int end_of_my_window = (id + 1) * template_length;

    if (overlap_recheck[id] == 0)
    {
        return;
    }

    best_spike_indices[id] = overlap_best_spike_indices[id];
    best_spike_labels[id] = overlap_best_spike_labels[id];
    if ((overlap_best_spike_indices[id] < start_of_my_window) && (id > 1))
    {
        if (overlap_recheck[id - 2] == 1)
        {
            /* To avoid situation where both units try to move into same window, */
            /* Arbitrarily have the window to the right (the current window) */
            /* give up and wait for a future iteration */
            /* Setting these to zero ensures we won't try to add later */
            overlap_recheck[id] = 0;
            best_spike_indices[id] = 0;
            /* But we should still recheck its neighbors as if it were added */
            check_window_on_next_pass[id - 1] = 2;
            check_window_on_next_pass[id] = 2;
            check_window_on_next_pass[id + 1] = 2;
        }
        else
        {
            check_window_on_next_pass[id - 2] = 2;
        }
    }
    if ((overlap_best_spike_indices[id] >= end_of_my_window) && (id + 2 < voltage_length / template_length))
    {
        check_window_on_next_pass[id + 2] = 2;
    }
    return;
}


/**
 * Kernel used to perform binary pursuit. The value of each residual voltage is checked
 * to see if we should add a spike of the passed templates. This kernel is
 * designed to be called multiple times until num_additional_spikes does not change.
 *
 * Each kernel is assigned a window whose length is equal to template_length. The kernel
 * checks the preceeding window and the next window to determine if a spike should
 * be added in any of these three windows. If a spike should be added in the current
 * kernel's window, the spike is added to the output and subtracted from voltage. However,
 * if either the preceeding window or the next window show a larger maximum likelihood,
 * we return without doing anytime (believing that a seperate kerel will take care
 * of those spikes).
 *
 * It is possible that, if the maximum likelihood spike, occurs another window, we
 * may still need to add a spike in our current window after the other kernel takes
 * care of the spike in the preceeding or succeeding window. Therefore, this kernel
 * must be run multiple times until num_additional_spikes == 0.
 *
 * Parameters:
 *  voltage: A float32 vector containing the voltage data
 *  voltage_length: The length of voltage
 *  num_neighbor_channels: The number of neighbors channels packed into voltage and templates
 *  templates: A num_timepoints * num_channels * num_neurons vector (see above description)
 *  num_templates: The dimension M of templates (num_neurons)
 *  template_length: The dimension N of templates (num timepoints)
 *  template_sum_squared: A 1xnum_templates vector containing the template sum squared values
 *  gamma: 1xnum_templates vector containing the bias parameter
 *  window_indices: A list of window numbers (0, 1, 2, 3,... each referring to a non-overlapping
 *   window of width template_length. If this is null, we use the global id. See the
 *   function compute_template_maximum_likelihood for more documentation.
 *  num_window_indices: The length of the window indices vector. If this value is zero, we
 *   use the global id as our window index.
 *  local_scatch: 1xnum_local_workers vector for computing the prefix_sum
 *  num_additional_spikes [out]: A global integer representing the number of additional spikes we have added
 *  additional_spike_indices [out]: A global vector (max length = 1xnum_workers) containing the indices within voltage
 *   where we have added spikes. Only 1:num_additional_spikes are valid (indices after num_additiona_spikes
 *   are garbage/don't care).
 *  additional_spike_labels [out]: The template ids that we have added at additional spike_indices. Same
 *   restrictions as additional_spike_indices.
 */
__kernel void binary_pursuit(
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    const unsigned int template_length,
    __global const float * restrict likelihood_lower_thresholds,
    __global unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global const unsigned int * restrict best_spike_indices,
    __global const unsigned int * restrict best_spike_labels,
    __global const float * restrict best_spike_likelihoods,
    __local unsigned int * restrict local_scratch,
    __global unsigned int * restrict num_additional_spikes,
    __global unsigned int * restrict additional_spike_indices,
    __global unsigned int * restrict additional_spike_labels,
    __global unsigned char * restrict overlap_recheck,
    __global unsigned char * restrict check_window_on_next_pass)
{
    const size_t global_id = get_global_id(0);
    const size_t local_id = get_local_id(0);
    const size_t local_size = get_local_size(0);
    size_t id;
    if (num_window_indices > 0 && window_indices != NULL && global_id >= num_window_indices)
    {
        /* Extra worker with nothing to do */
        /* To ensure this doesn't do any work, we set the id beyond hte length of voltage */
        id = (size_t) (voltage_length / template_length) + 1;
    }
    else if (num_window_indices == 0 || window_indices == NULL)
    {
        /* No window id's passed, just use our global id */
        id = global_id;
    }
    else
    {
        /* Used our passed window indices */
        id = window_indices[global_id];
    }

    /* Only spikes found within [id * template_length, id + 1 * template_length] are added to the output */
    /* All other spikes are ignored (previous window and next window) */
    const unsigned int start_of_my_window = ((signed int) id) * ((signed int) template_length);
    const unsigned int end_of_my_window = (id + 1) * template_length;

    /* Define our private variables */
    unsigned int maximum_likelihood_neuron = 0;
    float maximum_likelihood = 0.0;
    unsigned int maximum_likelihood_index = 0;
    unsigned int has_spike = 0; /* Does the current worker have a spike ? */

    /* Define a small local variable for our global offset */
    __local unsigned int global_offset;


    if (end_of_my_window < voltage_length - template_length && num_neighbor_channels > 0)
    {
        maximum_likelihood = best_spike_likelihoods[id];
        maximum_likelihood_neuron = best_spike_labels[id];
        maximum_likelihood_index = best_spike_indices[id];
    }

    /* If the best maximum likelihood is greater than threshold and within our window */
    /* or was an overlap recheck) add the spike to the output */
    local_scratch[local_id] = 0;
    has_spike = 0;
    if ( (maximum_likelihood > likelihood_lower_thresholds[maximum_likelihood_neuron]) )
    {
        if ( ((maximum_likelihood_index >= start_of_my_window) && (maximum_likelihood_index < end_of_my_window))
            || (overlap_recheck[id] == 1) )
        {
            local_scratch[local_id] = 1;
            has_spike = 1;
            check_window_on_next_pass[id] = 3;
            if (id > 0)
            {
                check_window_on_next_pass[id-1] = 3;
            }
            if (id + 1 < voltage_length / template_length)
            {
                check_window_on_next_pass[id+1] = 3;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for all workers to get here */
    prefix_local_sum(local_scratch); /* Compute the prefix sum to give our offset into spike indices) */

    /* The purpose of this is to increase the convergence rate. Without it the
    algorithm will repeatedly check windows flagged in 'compute_template_maximum_likelihood'
    even if they do not cross threshold and will never be added. These events have
    check_window_on_next_pass[id] == 1, whereas windows flagged for recheck for
    other reasons are assigned check_window_on_next_pass[id] > 1. */
    if ( (end_of_my_window < voltage_length - template_length && num_neighbor_channels > 0)
        && (has_spike == 0) )
    {
        /* This check massively increases convergence across the number of windows
        to check on each iteration. It tests for the situation where the maximum
        likelihood occurs in a neighboring window and is over threshold, but is
        blocked from being added by a sub threshold event with greater likelihood
        in a neighboring window 2+ IDs away from the current one. Such blocking
        can daisy chain and create a situation where many windows are checked on
        each iteration without the chance of a spike ever being added.
        This might be able to create a race situation but I don't think it will
        affect the outcome. Just might converge in a different
        number of iterations but the rule is still correct because the other
        check for convergence below happens after this and after a barrier.
        */
        if ( (maximum_likelihood_index < start_of_my_window)
            && (check_window_on_next_pass[id] == 1) && (id > 0) )
        {
            if (check_window_on_next_pass[id-1] == 0)
            {
                check_window_on_next_pass[id] = 0;
            }
        }
        if ( (maximum_likelihood_index >= end_of_my_window)
            && (check_window_on_next_pass[id] == 1) && (id + 1 < voltage_length / template_length) )
        {
            if (check_window_on_next_pass[id+1] == 0)
            {
                check_window_on_next_pass[id] = 0;
            }
        }
        if ( (maximum_likelihood <= likelihood_lower_thresholds[maximum_likelihood_neuron])
            && (check_window_on_next_pass[id] < 3) )
        {
            check_window_on_next_pass[id] = 0;
        }
    }

    if ((local_id == (local_size - 1)) && (local_scratch[(local_size - 1)] || has_spike)) /* I am the last worker and there are spikes to add in this group */
    {
        /* NOTE: The old value of num_additional spikes is returned as global offset */
        /* This is an atomic add on a global value (sync'ed across work groups) */
        global_offset = atomic_add(num_additional_spikes, (local_scratch[(local_size - 1)] + has_spike));
    }
    barrier(CLK_LOCAL_MEM_FENCE); /* Ensure all local workers get the same global offset */

    /* Each local worker stores their result in the appropriate offset */
    const unsigned int my_offset = global_offset + local_scratch[local_id];
    if (has_spike)
    {
        additional_spike_indices[my_offset] = maximum_likelihood_index;
        additional_spike_labels[my_offset] = maximum_likelihood_neuron;
    }
    return;
}

/**
 * Kernel used to obtain the clips across channels with the effects of nearby
 * spiking (from other neurons) removed from each clip.
 *
 * This kernel is effectively the reverse of the compute residuals kernel, above.
 * Given a residual voltage, with the effect of all spiking removed, we return
 * the clips across electrodes (which is just the residual voltage vector +
 * the respective template at the given spike index).
 *
 * Parameters:
 *  voltage: A float32 vector containing the voltage data
 *  voltage_length: The length of voltage for a single channel
 *  num_neighbor_channels: The number of neighbors channels packed into voltage and templates
 *  templates: A num_timepoints * num_channels * num_neurons vector (see above description)
 *  num_templates: The dimension M of templates (num_neurons)
 *  template_length: The dimension N of templates (num timepoints)
 *  spike_indices: A vector containing the initial index of each spike to remove. Each
 *   index is the FIRST value where we should subtract the first value of the template.
 *  spike_label: The label (starting at 0 NOT 1) for each spike as an index into templates.
 *  spike_indices_length: The number of spike indices (and length of spike labels).
 *  clips: A 1x(spike_indices*template_length*num_neighbor_channels) vector for the output
 *   clips with other neuron spiking removed.
 */
__kernel void get_adjusted_clips(
    __global const voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    __global const unsigned int * spike_indices,
    __global const unsigned int * spike_labels,
    const unsigned int spike_indices_length,
    __global voltage_type * restrict clips)
{
    const size_t id = get_global_id(0);
    /* Return if we started a kernel and it has nothing to do */
    if (id >= spike_indices_length || num_neighbor_channels == 0)
    {
        return; /* Do nothing */
    }

    unsigned int i, current_channel;
    const unsigned int spike_index = spike_indices[id]; /* The first index of our spike */
    const unsigned int spike_label = spike_labels[id]; /* Our template label (row in templates) */
    if (spike_label >= num_templates)
    {
        return; /* this is technically an error. This label does not exist in templates */
    }

    for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
    {
        unsigned int voltage_offset = spike_index + (voltage_length * current_channel); /* offset in voltage for this channel channel */
        unsigned int template_offset = (spike_label * template_length * num_neighbor_channels) + (current_channel * template_length);
        unsigned int clip_offset = (id * template_length * num_neighbor_channels) + (current_channel * template_length);
        for (i = 0; i < template_length; i++)
        {
            if (spike_index + i >= voltage_length)
            {
                break;
            }
            clips[clip_offset + i] = (voltage_type) (voltage[voltage_offset + i] + templates[template_offset + i]);
        }
    }
    return;
}
