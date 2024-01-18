from algorithms.online_kcenter import color_centerer
from algorithms.utils import Stopwatch
from fmmdmwu_nyoom import epsilon_falloff as FMMDMWU

def fmmdmwu_stream(gen, features, colors, kis, gamma_upper, mwu_epsilon, falloff_epsilon, return_unadjusted, percent_theoretical_limit=1.0, streamtimes = False, otherdmax = 0.0, useOtherDmax= False):
    
    # compute final k value
    k = sum(kis.values())

    (_, dim) = features.shape

    # make streamed coreset of size k*m
    timer = Stopwatch("Finalize centers")
    centerer = color_centerer(k)
    centerer.add(features, colors)
    core_features, core_colors = centerer.get_centers()
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
    
    print('********StreamMFD Param stats**********')
    print(f'\t dmax(stream) = {dmax}')
    core_stats = {}
    for i in range(0, len(core_features)):
        if core_colors[i] in core_stats:
            core_stats[core_colors[i]] += 1
        else:
            core_stats[core_colors[i]] = 1
    print(f'\t core stats:')
    for iter in core_stats:
        print(f'\t\t {iter} : {core_stats[iter]}')
    print('********StreamMFD Param stats**********')


    # Run MWU on the calculated coreset
    if useOtherDmax:
        dmax = otherdmax
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

    from algorithms.utils import check_returned_kis
    kis_delta = check_returned_kis(core_colors, kimap, sol)
    print('********StreamMFD KIS DELTA**********')
    for iter in kis_delta:
        print(f'\t\t {iter} : {kis_delta[iter]}')
    print('********StreamMFD KIS DELTA**********')



    if streamtimes:
        return sol, div, [avg_point_t, t_alg + last_finalize_time + dmax_compute_time, total_time]
    else:
        return sol, div, total_time


