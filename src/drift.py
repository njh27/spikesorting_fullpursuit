
using Statistics

"""
    stitch_segments(clips, clips_across_electrodes, segment_crossings, segment_labels)

Stitch multiple segments together, keeping track of the unique neurons across segments.
We join two neurons from adjacent segments if their templates match within a certain
threshold - if so these are considered the same neuron.

The goal of this function is to account for slow-drift in the electrode across time (segments)
where the shape of the waveform of a given neuron may change gradually.
"""
function stitch_segments(clips::AbstractVector{<:AbstractArray{<:Integer}}, clips_across_electrodes::AbstractVector{<:AbstractArray{<:Integer}}, segment_crossings::AbstractVector{<:AbstractVector{<:Integer}}, segment_labels::AbstractVector{<:AbstractVector{<:Integer}}; threshold::Real=0.05, verbose::Bool=false)
    if length(segment_crossings) == 1
        return clips[1], clips_across_electrodes[1], segment_crossings[1], segment_labels[1] # Nothing to stitch
    end

    # Remove any empty segments from clips
    remove = [length(segment) == 0 for segment in segment_crossings]
    deleteat!(clips, remove)
    deleteat!(clips_across_electrodes, remove)
    deleteat!(segment_crossings, remove)
    deleteat!(segment_labels, remove)

    distance_matrix = compute_segment_distance_matrix(clips)
    # Find 2-opt optimal path to minimize adjacent distances (traveling salesman)
    #I, _ = two_opt_path(distance_matrix)
    I = 1:length(segment_crossings)

    # Reorder segments by the minimum distance route
    segment_crossings = segment_crossings[I]
    segment_labels = segment_labels[I]
    clips = clips[I]

    # Keep track of the number of unique labels across
    # our complete recording.
    reorder_labels!(segment_labels[1])
    num_unique_labels = length(unique(segment_labels[1]))

    for i = 2:length(segment_crossings)
        previous_unique_labels = sort(unique(segment_labels[i-1]))
        previous_templates, _ = compute_templates(clips[i-1], segment_labels[i-1], previous_unique_labels)

        reorder_labels!(segment_labels[i])
        segment_labels[i] .= segment_labels[i] .+ num_unique_labels # Ensures uniqueness across these two segments

        current_unique_labels = sort(unique(segment_labels[i]))
        current_templates, _ = compute_templates(clips[i], segment_labels[i], current_unique_labels)
        # Assume that all found neurons in the current segment will have no match with a
        # previous segment neuron. Therefore, the total number of unique labels at this
        # point is equal to number of previously unique labels plus the currently found
        # neurons.
        num_unique_labels = num_unique_labels + length(current_unique_labels)

        verbose ? println("\t\tWorking on segment ", i, " of ", length(segment_crossings), ", with ", length(current_unique_labels), " neurons.") : nothing

        for j = 1:length(current_unique_labels)
            current_select = segment_labels[i] .== current_unique_labels[j]
            match_found = false
            for k = 1:length(previous_unique_labels)
                previous_select = segment_labels[i-1] .== previous_unique_labels[k]
                current_labels = vcat(zeros(Int64, sum(previous_select)), ones(Int64, sum(current_select)))
                current_clips = vcat(clips[i-1][previous_select, :], clips[i][current_select, :])
                merge_clusters!(current_clips, current_labels)

                if length(unique(current_labels)) == 1
                    # This is the same as the previously found neuron. Change the label
                    # so that it is the same as the previous neuron
                    segment_labels[i][segment_labels[i] .== current_unique_labels[j]] .= previous_unique_labels[k]
                    num_unique_labels -= 1
                    verbose ? println("\t\tFound match between previous segment neuron ", k, " and current neuron ", j) : nothing
                    match_found = true
                    break
                end
            end
            if (! match_found)
                verbose ? println("\t\tNO match between any previous segment neurons and current neuron ", j) : nothing
            end
        end
        verbose ? println("\t\tDone with segment ", i) : nothing
    end

    # Restore the segment crossings to their original order
    crossings = vcat(segment_crossings...)
    labels = vcat(segment_labels...)
    clips = vcat(clips...)
    clips_across_electrodes = vcat(clips_across_electrodes...)

    I = sortperm(crossings)
    crossings = crossings[I]
    labels = labels[I]
    clips = clips[I, :]
    clips_across_electrodes = clips_across_electrodes[I, :]
    reorder_labels!(labels) # Ensure labels go from 1:num_unique_labels

    return clips, clips_across_electrodes, crossings, labels
end


"""
    compute_segment_distance_matrix(clips)

Computes and returns the NxN distance matrix (where N is the length of clips)
based on the Forbenius norm distance. We calculate this distance matrix by taking
the PCA across all of the clips and then projecting each clip on to these PCA components.
Each segment forms an M-D point in space, which is the average projection onto these
PCAs. The distance between any two segments is given by a modification of the
Forbenius norm. For instance, imagine the vector `A' is the 1 x num_pca_components
vector for segment 2. `B' is the 1 x num_pca_components matrix for segment 2.
    distance = sqrt((A - B) * (A - B)^T)
"""
function compute_segment_distance_matrix(clips::AbstractVector{<:AbstractArray{<:Integer}}; max_components::Integer=20)
    # Get the PCA components across all clips
    components = pca(vcat(clips...), max_components)

    distance_matrix = zeros(length(clips), length(clips))
    scores = zeros(length(clips), size(components, 2))

    for i = 1:length(clips)
        current_scores = pca_scores(clips[i], components)
        scores[i, :] = dropdims(mean(current_scores, dims=1), dims=1) # Average across spikes
    end

    for i = 1:length(clips)
        for j = i:length(clips)
            distance = sqrt((scores[i, :] .- scores[j, :])' * (scores[i, :] .- scores[j, :]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
        end
    end

    return distance_matrix
end

"""
    closest_neighbor_path(distances, start_node)

This function uses the clostest/nearest neighbor heuristic to solve the
traveling salesman problem. Note that the problem salesman problem is NP-hard,
so it would take N! permutations to find the true optimal path that minimizes
the overall distance. We use an approximation of the optimal path by choosing
the closest node (city) from each city and proceeding until there are no nodes
(cities) left to visit. The resulting path is returned along with the cummulative
cost for the entire trip.
"""
function closest_neighbor_path(distances::AbstractArray{<:Real}, starting_node::Integer)
    # A complete set of vertices that have not been visted
    vertex_set = collect(1:size(distances, 1))
    shortest_path_set = Vector{Int64}(undef, 0)
    adjacent_distances = Inf .* ones(length(vertex_set))
    adjacent_distances[starting_node] = distances[starting_node, starting_node]
    total_cost = 0

    while length(vertex_set) != 0
        minimum_cost, index = findmin(adjacent_distances)
        total_cost = total_cost + minimum_cost

        # Add our index onto the shortest path
        push!(shortest_path_set, vertex_set[index])
        # Delete that index from our set of non-visted nodes
        deleteat!(vertex_set, index)
        # Update our distances
        adjacent_distances = distances[shortest_path_set[end], vertex_set]
    end

    return shortest_path_set, total_cost
end

"""
    closest_neighbor_path(distances)

Takes a distance matrix and compute the optimal path where any node
can be the start node. In essence, this function just calls
closest_neighbor_path(distances, start_node) for every node in the
list and then returns the shortest (least costly) overall path.
"""
function closest_neighbor_path(distances::AbstractArray{<:Real})
    minimum_path = nothing
    minimum_cost = Inf
    for i = 1:size(distances, 2)
        I, cost = closest_neighbor_path(distances, i)
        if cost < minimum_cost
            minimum_cost = cost
            minimum_path = I
        end
    end
    return minimum_path, minimum_cost
end


"""
    two_opt_path(distances)

Using the passed distance matrix, compute the 2-opt optimal path. We get here
by first creating the optimal path using the nearest neighbor algorithm. This
algorithm has the potential to have "crossings" in the path, resulting in a
sub-optimal total path length. We take this path and try every permutation of
swapping a vertex to see if it reduces our overall path length. The resulting
path is considered two-opt optimal.

For more information, see:
    https://en.wikipedia.org/wiki/2-opt

This function returns the two-opt path as well as the cost for the complete path.
"""
function two_opt_path(distances::AbstractArray{<:Real})
    function cost_difference(distances::AbstractArray{<:Real}, route::Vector{<:Integer}, i::Integer, j::Integer)
        # Check distance between points (A, B) + (C, D) against (A, C) + (B, D)
        return (distances[route[i], route[j+1]] + distances[route[i-1], route[j]]) - (distances[route[i], route[i-1]] + distances[route[j+1], route[j]])
    end
    function swap_route(route::Vector{<:Integer}, i::Integer, j::Integer)
        return vcat(route[1:i-1], reverse(route[i:j]), route[j+1:end]) # TODO: Fix indices
    end
    route, cost = closest_neighbor_path(distances)

    improved = true
    while improved
        improved = false
        for i = 2:(length(route) - 2)
            for j = (i + 1):length(route) - 1
                difference = cost_difference(distances, route, i, j)

                if difference < 0
                    cost = cost + difference
                    # Swap route
                    route = swap_route(route, i, j)
                    improved = true
                end
            end
        end
    end

    return route, cost
end
