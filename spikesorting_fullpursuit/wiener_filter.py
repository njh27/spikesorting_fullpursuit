import numpy as np



def wiener_filter_segment():
    """ Does the Wiener filter on the segment voltage provided. """

settings[:wiener_filter] = true
    settings[:wiener_filter_smoothing] = 100 # Hz or nothing for no smoothing
segment_probe = nothing
            if sorter_settings[:binary_pursuit] && sorter_settings[:wiener_filter]
                wiener_filter_smooth_indices=div(sorter_settings[:wiener_filter_smoothing] * (end_index - start_index + 1), sampling_rate(probe[1][sorter_settings[:timeseries_type]]) // 2)
                segment_probe = wiener_filter_segment(probe,
                    neuron_channels,
                    neuron_crossings,
                    neuron_templates,
                    timeseries_type=sorter_settings[:timeseries_type],
                    start_index=start_index,
                    end_index=end_index,
                    clip_size=sorter_settings[:clip_size],
                    smooth=wiener_filter_smooth_indices)
                # Ensure neuron templates are relative to the start of this new recording
                neuron_crossings = [crossings .- start_index .+ 1 for crossings in neuron_crossings]
                # Recompute the templates now that we have filtered
                neuron_templates = [compute_template(get_clips_across_electrodes(segment_probe, neuron_crossings[i], neighbors(probe, neuron_channels[i]), clip_size=sorter_settings[:clip_size])) for i = 1:length(neuron_templates)]
            end


"""
    wiener(input, signal, noise)

Compute the wiener filter on the input signal given an estimate of the
true (noiseless) signal and an estimate of the noise (signal removed).

This function optimally smooths the noise and signal power spectra using
a running average specified by `smooth`. The smoth parameter is provided
in integer samples (i.e., the boxcar window).
"""
function wiener(input::AbstractVector{<:Real}, signal::AbstractVector{<:Real}, noise::AbstractVector{<:Real}; smooth::Union{Integer, Nothing}=nothing)
    @assert length(input) >= length(signal) && length(input) >= length(noise)
    # Zero pad noise and signal
    if length(signal) < length(input)
        signal = vcat(signal, zeros(eltype(signal), length(input) - length(signal)))
    end
    if length(noise) < length(input)
        noise = vcat(noise, zeros(eltype(noise), length(input) - length(noise)))
    end

    input_ft = rfft(input) # Input in frequency domain
    S = abs2.(rfft(signal)) # Signal power spectrum
    N = abs2.(rfft(noise)) # Noise power spectrum
    if smooth != nothing && smooth > 1
        # Rolling sum implementation of box-car filter separately of N
        # and S using the binsize passed in the `smooth` variable.
        S_smoothed = similar(S)
        N_smoothed = similar(N)
        half_bin_size = div(smooth, 2)
        rolling_sum_num_points = min(half_bin_size, length(N))
        N_rolling_sum = sum(N[1:rolling_sum_num_points])
        S_rolling_sum = sum(S[1:rolling_sum_num_points])
        for i = 1:length(S)
            if i + half_bin_size < length(S)
                N_rolling_sum += N[i+half_bin_size]
                S_rolling_sum += S[i+half_bin_size]
                rolling_sum_num_points += 1
            end
            if i - half_bin_size > 0
                N_rolling_sum -= N[i-half_bin_size]
                S_rolling_sum -= S[i-half_bin_size]
                rolling_sum_num_points -= 1
            end
            S_smoothed[i] = S_rolling_sum / rolling_sum_num_points
            N_smoothed[i] = N_rolling_sum / rolling_sum_num_points
        end
        S = S_smoothed
        N = N_smoothed
    end

    filtered_signal = irfft(input_ft .* wiener_optimal_filter(S, N), length(input))
    return filtered_signal
end


function wiener_optimal_filter(signal_power::AbstractVector{<:Real}, noise_power::AbstractVector{<:Real}; epsilon::Real=1e-9)
    @assert length(signal_power) == length(noise_power)
    # Without blurring, the filter is:
    #   |S|² / (|S|² + |N|²)
    return (signal_power) ./ (signal_power .+ noise_power .+ epsilon)
end


"""
    wiener_filter_segment(probe, templates, crossings)

Create an InMemoryRecording for the passed segment where the resulting voltage
is filtered by the wiener filter. A probe object with the filtered signal
is returned (the original probe is not altered).
"""
function wiener_filter_segment(probe::AbstractProbe,
        channels::AbstractVector{<:Integer},
        crossings::AbstractVector{<:AbstractVector{<:Integer}},
        templates::AbstractVector{<:AbstractVector{<:Real}};
        timeseries_type::Type{<:AbstractNeurophysiologyChannelTimeseriesType}=SpikeTimeseries,
        start_index::Integer=0,
        end_index::Integer=length(recording[1][timeseries_type]),
        clip_size::Union{Real, AbstractVector}=1e-3,
        kwargs...)

    # Calculate the number of pre-indices for our template
    if isa(clip_size, Real)
        pre_indices = Integer(round(clip_size / 2.0 / dt(probe[1][timeseries_type])))
    else
        pre_indices = Integer(round(clip_size[1] / dt(probe[1][timeseries_type])))
    end

    # Create a new voltage matrix that we will use to construct our final in memory probe
    segment_voltage = zeros(eltype(probe[1][timeseries_type]), length(probe), end_index - start_index + 1)
    # Fill it with our measured voltage
    for channel_index = 1:length(probe)
        segment_voltage[channel_index, :] .= probe[channel_index][timeseries_type][start_index:end_index]

        noiseless_signal = zeros(eltype(segment_voltage), size(segment_voltage, 2))
        noise = probe[channel_index][timeseries_type][start_index:end_index]

        for neuron_index = 1:length(templates)
            # Get our template on this channel from the template
            if ! (channel_index in neighbors(probe, channels[neuron_index]))
                continue
            end
            channel_template = compute_subtemplate(templates[neuron_index], neighbors(probe, channels[neuron_index]), [channel_index])
            channel_template = convert.(eltype(segment_voltage), round.(channel_template))
            for index = crossings[neuron_index]
                index = index - pre_indices - start_index + 1
                if index < 1 || index + length(channel_template) > size(segment_voltage, 2)
                    continue
                end
                noiseless_signal[index:index+length(channel_template)-1] .+= channel_template
                noise[index:index+length(channel_template)-1] .-= channel_template
            end
        end

        filtered_signal = wiener(segment_voltage[channel_index, :], noiseless_signal, noise; kwargs...)
        filtered_signal = filtered_signal .* std(segment_voltage[channel_index, :]) ./ std(filtered_signal)
        if any(isnan.(filtered_signal))
            @warn "Wiener filter failed for channel $channel_index due to the presence of NaNs. Skipping."
            continue
        end
        filtered_signal = convert.(eltype(segment_voltage), clamp.(round.(filtered_signal), typemin(eltype(segment_voltage)), typemax(eltype(segment_voltage)))) # Clamp does conversion automatically
        segment_voltage[channel_index, :] .= filtered_signal
    end
    # Now that we have our filtered signal, convert this into our probe object
    segment_recording = InMemoryRecording(segment_voltage, sampling_rate(probe[1][timeseries_type]), timeseries_type=timeseries_type)
    segment_probe = ProbeContainer(probe, segment_recording)
    return segment_probe
end
