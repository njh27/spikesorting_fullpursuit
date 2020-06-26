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
    __global const float * restrict template_sum_squared,
    __global const float * restrict gamma)
{
    unsigned int i;
    unsigned int current_channel;
    /* Set our maximum likelihood found to 0.0 (no spike to add) */
    float maximum_likelihood = 0.0;
    if (index + template_length >= voltage_length)
    {
        return maximum_likelihood; /* Returns maximum likelihood = 0.0 */
    }
    if (template_number >= template_length)
    {
        return maximum_likelihood; /* Invalid template number, return 0.0 */
    }
    maximum_likelihood = template_sum_squared[template_number] - gamma[template_number];

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
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    const unsigned int template_number,
    __global const float * restrict template_sum_squared,
    __global const float * restrict gamma,
    __global const unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global float * restrict best_spike_likelihoods,
    __global unsigned char * restrict check_window_on_next_pass,
    __global unsigned char * restrict overlap_recheck)
{
    const size_t global_id = get_global_id(0);
    if (num_window_indices > 0 && window_indices != NULL && global_id >= num_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    const size_t id = (num_window_indices > 0 && window_indices != NULL) ? window_indices[global_id] : global_id;
    unsigned int i;
    unsigned int check_window = 0;

    /* Only spikes found within [id * template_length, id + 1 * template_length] are added to the output */
    /* All other spikes are ignored (previous window and next window) */
    const unsigned int start_of_my_window = ((signed int) id) * ((signed int) template_length);
    const unsigned int end_of_my_window = (id + 1) * template_length - 1;
    const unsigned int start = (template_length > start_of_my_window) ? 0 : (start_of_my_window - template_length);
    const unsigned int stop = (end_of_my_window + template_length) > voltage_length ? voltage_length : (end_of_my_window + template_length);

    if (template_number >= template_length)
    {
        return; /* Invalid template number */
    }

    if (num_neighbor_channels == 0)
    {
        return; /* Invalid number of channels (must be >= 1) */
    }
    if (end_of_my_window >= voltage_length - template_length)
    {
        return; /* Nothing to do, the end of the current window exceeds the bounds of voltage */
    }

    /* Cache the data from the global buffers so that we only have to write to them once */
    __private float best_spike_likelihood_private = best_spike_likelihoods[id];
    __private unsigned int best_spike_label_private = best_spike_labels[id];
    __private unsigned int best_spike_index_private = best_spike_indices[id];
    __private float raw_likelihood;

    if (template_number == 0)
    {
        /* Reset our best_spike_likelihoods*/
        best_spike_likelihood_private = 0.0;
    }
    /* Compute our minimum cost to go for adding our current template at each point in the window */
    for (i = 0; i < (unsigned) stop - start; i++)
    {
        float current_maximum_likelihood = compute_maximum_likelihood(voltage, voltage_length, num_neighbor_channels,
            i + start, templates, num_templates, template_length, template_number,
            template_sum_squared, gamma);
        if (current_maximum_likelihood > best_spike_likelihood_private)
        {
            best_spike_likelihood_private = current_maximum_likelihood;
            best_spike_label_private = template_number;
            best_spike_index_private = start + i;
        }
        if ((current_maximum_likelihood > 0.0) && (start + i >= start_of_my_window) && (start + i <= end_of_my_window))
        {
            check_window = 1;
        }
    }
    if (check_window && check_window_on_next_pass != NULL)
    {
        check_window_on_next_pass[id] = 1;
        if (best_spike_index_private >= start_of_my_window && best_spike_index_private <= end_of_my_window)
        {
            if (id > 0)
            {
                check_window_on_next_pass[id-1] = 1;
            }
            if (id + 1 < voltage_length / template_length)
            {
                check_window_on_next_pass[id+1] = 1;
            }
        }
    }
    if ((best_spike_likelihood_private > 0.0) && (best_spike_index_private >= start_of_my_window) && (best_spike_index_private <= end_of_my_window))
    {
        /* Best spike is in current window so check whether it violates its expected delta likelihood */
        /* If yes, flag this spike for recheck, else set recheck back to zero */
        raw_likelihood = best_spike_likelihood_private + gamma[best_spike_label_private];
        if ((raw_likelihood < -1*template_sum_squared[best_spike_label_private] - gamma[best_spike_label_private])
            || (raw_likelihood > -1*template_sum_squared[best_spike_label_private] + gamma[best_spike_label_private]))
        {
            overlap_recheck[id] = 1;
        }
        else
        {
            overlap_recheck[id] = 0;
        }
    }

    /* Write our results back to the global vectors */
    best_spike_likelihoods[id] = best_spike_likelihood_private;
    best_spike_labels[id] = best_spike_label_private;
    best_spike_indices[id] = best_spike_index_private;
    return;
}

/* Kernel chooses the current best_spike_label for its window ID and fixes its
placement at the index best_spike_indices[best_spike_label] + fixed_shift_index.
It then proceeds to compute the likelihood for the combination of this best unit
at its new shifted fixed index, and the unit corresponding to template_number
for all possible shifts +/- n_shift points relative to the fixed shift index.
If the likelihood is improved over the current best likelihood in this window,
the best_spike_likelihood for this window is updated, and best_spike_indices for
this window is assigned as best_spike_indices[best_spike_label] + fixed_shift_index.
The label for the best unit is unchanged. The test template indicated by template
number and its optimal shift generating the peak likelihood are disregarded.
Must be followed by a call to check_overlap_reassignments to guaruntee that
indices are able to be added without interferring with other spikes.
*/
__kernel void overlap_recheck_indices(
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    const unsigned int template_number,
    const signed int fixed_shift_index,
    const unsigned int n_shift_points,
    __global const float * restrict template_sum_squared,
    __global const float * restrict gamma,
    __global const unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global unsigned int * restrict best_spike_labels,
    __global float * restrict best_spike_likelihoods)
{
    const size_t global_id = get_global_id(0);
    if (num_window_indices > 0 && window_indices != NULL && global_id >= num_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    const size_t id = (num_window_indices > 0 && window_indices != NULL) ? window_indices[global_id] : global_id;
    unsigned int i;
    unsigned int j;
    unsigned int current_channel;

    /* Only spikes found within [id * template_length, id + 1 * template_length] are added to the output */
    /* All other spikes are ignored (previous window and next window) */
    // const unsigned int start_of_my_window = ((signed int) id) * ((signed int) template_length);
    const unsigned int end_of_my_window = (id + 1) * template_length - 1;
    // const unsigned int start = (template_length > start_of_my_window) ? 0 : (start_of_my_window - template_length);
    // const unsigned int stop = (end_of_my_window + template_length) > voltage_length ? voltage_length : (end_of_my_window + template_length);

    if (num_neighbor_channels == 0)
    {
        return; /* Invalid number of channels (must be >= 1) */
    }
    if (end_of_my_window >= voltage_length - template_length)
    {
        return; /* Nothing to do, the end of the current window exceeds the bounds of voltage */
    }

    /* Cache the data from the global buffers so that we only have to write to them once */
    __private float best_spike_likelihood_private = best_spike_likelihoods[id];
    __private unsigned int best_spike_label_private = best_spike_labels[id];
    __private unsigned int best_spike_index_private = best_spike_indices[id];

    if (((signed int) (best_spike_index_private + fixed_shift_index) < 0) || ((best_spike_index_private + fixed_shift_index) >= (voltage_length - template_length)))
    {
        return; // Fixed index is outside voltage range
    }

    __private float template_likelihood_at_index;
    __private float shifted_template_sse;
    __private float shift_sum;
    __private unsigned int absolute_fixed_index = best_spike_index_private + fixed_shift_index;
    __private unsigned int delta_index;



    /* Get likelihood for the current best spike label at the input fixed index relative to best index */
    template_likelihood_at_index = compute_maximum_likelihood(voltage, voltage_length, num_neighbor_channels,
        absolute_fixed_index, templates, num_templates, template_length, best_spike_label_private,
        template_sum_squared, gamma);
    /* Need to remove additive quantities to get appropriate distributivity of convolution */
    template_likelihood_at_index = template_likelihood_at_index - template_sum_squared[best_spike_label_private] + gamma[best_spike_label_private];

    /* Find absolute voltage indices we will check within shift range */
    const unsigned int shift_start = (n_shift_points > absolute_fixed_index) ? 0 : (absolute_fixed_index - n_shift_points);
    const unsigned int shift_stop = ((n_shift_points + absolute_fixed_index) > voltage_length) ? voltage_length : (absolute_fixed_index + n_shift_points);

    /* Compute the likelihood for adding the template given the position of the fixed best unit */
    for (i = 0; i < (unsigned) shift_stop - shift_start; i++)
    {
        float current_maximum_likelihood = compute_maximum_likelihood(voltage, voltage_length, num_neighbor_channels,
            i + shift_start, templates, num_templates, template_length, template_number,
            template_sum_squared, gamma);
        /* Need to remove additive quantities to get appropriate distributivity of convolution */
        current_maximum_likelihood = current_maximum_likelihood - template_sum_squared[template_number] + gamma[template_number];

        /* Compute the template sum squared for the combined templates at current shift */
        if ((i + shift_start) < absolute_fixed_index)
        {
            shifted_template_sse = 0.0;
            delta_index = absolute_fixed_index - (i + shift_start);
            for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
            {
                unsigned int template_offset = (template_number * template_length * num_neighbor_channels) + (current_channel * template_length);
                unsigned int fixed_template_offset = (best_spike_label_private * template_length * num_neighbor_channels) + (current_channel * template_length);
                for (j = 0; j < (delta_index + template_length); j++)
                {
                    /* Data only available for test template */
                    if (j < delta_index)
                    {
                        shifted_template_sse = shifted_template_sse + templates[template_offset + j] * templates[template_offset + j];
                    }
                    /* Data available for both templates */
                    if ((j >= delta_index) && (j < template_length))
                    {
                        shift_sum = templates[template_offset + j] + templates[fixed_template_offset + j - delta_index];
                        shifted_template_sse = shifted_template_sse + shift_sum * shift_sum;
                    }
                    /* Data only available for fixed template */
                    if (j >= template_length)
                    {
                        shifted_template_sse = shifted_template_sse + templates[fixed_template_offset + j - delta_index] * templates[fixed_template_offset + j - delta_index];
                    }
                }
            }
        }
        else
        {
            shifted_template_sse = 0.0;
            delta_index = (i + shift_start) - absolute_fixed_index;
            for (current_channel = 0; current_channel < num_neighbor_channels; current_channel++)
            {
                unsigned int template_offset = (template_number * template_length * num_neighbor_channels) + (current_channel * template_length);
                unsigned int fixed_template_offset = (best_spike_label_private * template_length * num_neighbor_channels) + (current_channel * template_length);
                for (j = 0; j < (delta_index + template_length); j++)
                {
                    /* Data only available for fixed template */
                    if (j < delta_index)
                    {
                        shifted_template_sse = shifted_template_sse + templates[fixed_template_offset + j] * templates[fixed_template_offset + j];
                    }
                    /* Data available for both templates */
                    if ((j >= delta_index) && (j < template_length))
                    {
                        shift_sum = templates[template_offset + j - delta_index] + templates[fixed_template_offset + j];
                        shifted_template_sse = shifted_template_sse + shift_sum * shift_sum;
                    }
                    /* Data only available for test template */
                    if (j >= template_length)
                    {
                        shifted_template_sse = shifted_template_sse + templates[template_offset + j - delta_index] * templates[template_offset + j - delta_index];
                    }
                }
            }
        }
        /* Use distributivity property of convolution to add likelihoods for fixed unit and test unit */
        current_maximum_likelihood = current_maximum_likelihood + template_likelihood_at_index;
        /* Correct current likelihood for the shifted template sse */
        current_maximum_likelihood = current_maximum_likelihood - 0.5 * shifted_template_sse;

        /* Current shifted likelihood beats previous best */
        if (current_maximum_likelihood > best_spike_likelihood_private)
        {
            /* Reset the likelihood and best index. Label is FIXED. */
            best_spike_likelihood_private = current_maximum_likelihood;
            best_spike_index_private = absolute_fixed_index;
        }
    }
    /* Write our results back to the global vectors */
    best_spike_likelihoods[id] = best_spike_likelihood_private;
    best_spike_labels[id] = best_spike_label_private;
    best_spike_indices[id] = best_spike_index_private;
    return;
}

/* If the new shifted index is outside
the current window for this worker, it will check whether a spike was added
+/- 2 windows away (the closest a spike could have been added), and if so, will
return, kicking the recheck can down the road until the next pass. */
__kernel void check_overlap_reassignments(
    const unsigned int template_length,
    __global const unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global unsigned int * restrict best_spike_indices,
    __global float * restrict best_spike_likelihoods,
    __global unsigned char * restrict check_window_on_next_pass,
    __global unsigned char * restrict overlap_recheck)
{
    const size_t global_id = get_global_id(0);
    if (num_window_indices > 0 && window_indices != NULL && global_id >= num_window_indices)
    {
        return; /* Extra worker with nothing to do */
    }
    const size_t id = (num_window_indices > 0 && window_indices != NULL) ? window_indices[global_id] : global_id;
    const unsigned int start_of_my_window = ((signed int) id) * ((signed int) template_length);
    const unsigned int end_of_my_window = (id + 1) * template_length - 1;

    /* Since overlap_recheck spikes are all maximum likelihood in their window, */
    /* we know there are no spikes in neighboring window. We instead must check */
    /* two windows away for interference if a best index was moved out of its */
    /* original window */
    /* NOTE: I am not sure this is guarunteed to allow convergence with the */
    /* policy of removing anything with a likelihood > 0.0 nearby. Whether */
    /* nearby units are also part of the overlap check must be considered. */
    /* NOTE: There is potential for a race condition with reassigning the */
    /* likelihoods here. I think since it is only in cases of a shifted window */
    /* it shouldn't actually matter */
    if (best_spike_indices[id] < start_of_my_window)
    {
        if (id > 1)
        {
            if (best_spike_likelihoods[id - 2] > 0.0)
            {
                /* Another spike is too close by to move to this index so kick the can down the road */
                best_spike_likelihoods[id] = 0.0;

            }
            /* NOTE: Do we need to add a policy for reassigning check_window_on_next_pass? */
            /* Main window has already been assigned as check windows as have their immediate neighbors */
            check_window_on_next_pass[id - 2] = 1;
        }
    }
    if (best_spike_indices[id] > end_of_my_window)
    {
        if (id < num_window_indices - 2)
        {
            /* Requiring that overlap_recheck == 0 allows convergence and enacts */
            /* the policy that in the event two neighbors are both rechecks, we */
            /* keep the one to the left */
            if ((best_spike_likelihoods[id + 2] > 0.0) && (overlap_recheck[id + 2] == 0))
            {
                /* Another spike is too close by to move to this index so kick the can down the road */
                best_spike_likelihoods[id] = 0.0;
            }
            /* NOTE: Do we need to add a policy for reassigning check_window_on_next_pass? */
            /* Main window has already been assigned as check windows as have their immediate neighbors */
            check_window_on_next_pass[id + 2] = 1;
        }
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
    __global voltage_type * restrict voltage,
    const unsigned int voltage_length,
    const unsigned int num_neighbor_channels,
    __global const voltage_type * restrict templates,
    const unsigned int num_templates,
    const unsigned int template_length,
    __global const float * restrict template_sum_squared,
    __global const float * restrict gamma,
    __global unsigned int * restrict window_indices,
    const unsigned int num_window_indices,
    __global const unsigned int * restrict best_spike_indices,
    __global const unsigned int * restrict best_spike_labels,
    __global const float * restrict best_spike_likelihoods,
    __local unsigned int * restrict local_scratch,
    __global unsigned int * restrict num_additional_spikes,
    __global unsigned int * restrict additional_spike_indices,
    __global unsigned int * restrict additional_spike_labels)
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
    const unsigned int end_of_my_window = (id + 1) * template_length - 1;

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
        if (maximum_likelihood > 0)
        {
            maximum_likelihood_neuron = best_spike_labels[id];
            maximum_likelihood_index = best_spike_indices[id];
        }
    }

    /* If the best maximum likelihood is greater than zero and within our window */
    /* add the spike to the output */
    if ((maximum_likelihood > 0.0) && (maximum_likelihood_index >= start_of_my_window) && (maximum_likelihood_index <= end_of_my_window))
    {
        local_scratch[local_id] = 1;
        has_spike = 1;
    }
    else
    {
        local_scratch[local_id] = 0;
        has_spike = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE); /* Wait for all workers to get here */
    prefix_local_sum(local_scratch); /* Compute the prefix sum to give our offset into spike indices) */

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
