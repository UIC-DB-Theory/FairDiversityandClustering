from algorithms.online_kcenter import color_centerer
from algorithms.utils import Stopwatch
from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU

def fmmdmwu_stream(gen, features, colors, kis, gamma_upper, mwu_epsilon, falloff_epsilon, return_unadjusted, percent_theoretical_limit=1.0, streamtimes = False):
    
    # compute final k value
    k = sum(kis.values())

    # get point dimension
    (_, dim) = features.shape

    # Stream the data
    """
    for feature, color in zip(features, colors):
        # TODO: Calculate the coreset for the streaming setting
        # Notes
        # Size of the coreset should be k*m, where m is the number of colors
        # For each color we run the clustering algorithm to get k points

        # Check if the bin has sufficient points
        if len(color_bins[color]) < k:
            # If not simply add the new point to the bin
            color_bins[color].append(feature)
        else:
            # TODO: Run k-center on set: color_bins[color] U {new point}
            pass
    
    # Merge the color bins to create the coreset
    core_features = []
    core_colors = []
    for color in color_bins:
        for feature in color_bins[color]:
            core_features.append(feature)
            core_colors.append(color)
    """
    # make streamed coreset of size k*m
    timer = Stopwatch("Finalize centers")
    centerer = color_centerer(k, dim)
    core_features, core_colors = centerer.add(features, colors)
    size = len(core_features)
    print(f'\t\tfirst point = {core_features[0]}, {core_colors[0]}')
    print(f'\t\tcoreset size (after stream) = {size}')
    avg_point_t, last_finalize_time = centerer.get_times()

    # Calculate dmax using coreset size k
    from algorithms.coreset import Coreset_FMM
    coreset = Coreset_FMM(
                gen,
                core_features, 
                core_colors, 
                k, 
                1,
                dim, 
                k)
    dmax = coreset.compute_gamma_upper_bound()
    print(f'\t\tdmax(stream) = {dmax}')
    dmax_compute_time = coreset.gamma_upper_bound_compute_time
    
    # Run MWU on the calculated coreset
    sol, div, t_alg = FMMDMWU(
        gen=gen,
        features = core_features, 
        colors = core_colors, 
        kis = kis,
        gamma_upper = dmax,
        mwu_epsilon = mwu_epsilon,
        falloff_epsilon = falloff_epsilon,
        percent_theoretical_limit = percent_theoretical_limit,
        return_unadjusted = return_unadjusted
    )
    _, total_time = timer.stop()

    if streamtimes:
        return sol, div, [avg_point_t, t_alg + last_finalize_time + dmax_compute_time, total_time]
    else:
        return sol, div, total_time


