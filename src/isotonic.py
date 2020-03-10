import numpy as np




def unimodal_prefix_isotonic_regression_l2(y, w):
    """ The y values must be ordered according to the desired fit, with the w
        values in one to one correspondence.  Weights w must be non-negative.
        """

    y_reverse = y[::-1]
    w_reverse = w[::-1]
    n = y.size
    error = np.zeros(n+1)
    err_asc = np.zeros(n)
    err_dec = np.zeros(n)
    y_new = np.copy(y)

    mean_vec = np.copy(y)
    sumwy_vec = y * w
    levelerr_vec = np.zeros(n)
    sumwy2_vec = sumwy_vec ** 2
    sumw_vec = np.copy(w)

    # Do ascending regression part getting the error
    vec_back = 0
    for j in range(1, n):
        err_asc[j] = err_asc[j - 1]
        while vec_back >= 0 and mean_vec[j] <= mean_vec[vec_back]:
            sumwy_vec[j] += sumwy_vec[vec_back]
            sumwy2_vec[j] += sumwy2_vec[vec_back]
            sumw_vec[j] += sumw_vec[vec_back]
            mean_vec[j] = sumwy_vec[j] / sumw_vec[j]
            levelerr_vec[j] = sumwy2_vec[j] - sumwy_vec[j] * sumwy_vec[j] / sumw_vec[j]
            err_asc[j] = err_asc[j] - levelerr_vec[vec_back]
            vec_back -= 1
        vec_back += 1
        sumwy_vec[vec_back] = sumwy_vec[j]
        sumwy2_vec[vec_back] = sumwy2_vec[j]
        sumw_vec[vec_back] = sumw_vec[j]
        mean_vec[vec_back] = mean_vec[j]
        levelerr_vec[vec_back] = levelerr_vec[j]
        err_asc[j] += levelerr_vec[vec_back]

    # Now do descending regression part, by first reversing our values
    mean_vec = np.copy(y_reverse)
    levelerr_vec = np.zeros(n)
    sumwy_vec = np.copy(y_reverse * w_reverse)
    sumwy2_vec = sumwy_vec ** 2
    sumw_vec = np.copy(w_reverse)

    vec_back = 0
    for j in range(1, n):
        err_dec[j] = err_dec[j - 1]
        while vec_back >= 0 and mean_vec[j] <= mean_vec[vec_back]:
            sumwy_vec[j] += sumwy_vec[vec_back]
            sumwy2_vec[j] += sumwy2_vec[vec_back]
            sumw_vec[j] += sumw_vec[vec_back]
            mean_vec[j] = sumwy_vec[j] / sumw_vec[j]
            levelerr_vec[j] = sumwy2_vec[j] - sumwy_vec[j] * sumwy_vec[j] / sumw_vec[j]
            err_dec[j] = err_dec[j] - levelerr_vec[vec_back]
            vec_back -= 1
        vec_back += 1
        sumwy_vec[vec_back] = sumwy_vec[j]
        sumwy2_vec[vec_back] = sumwy2_vec[j]
        sumw_vec[vec_back] = sumw_vec[j]
        mean_vec[vec_back] = mean_vec[j]
        levelerr_vec[vec_back] = levelerr_vec[j]
        err_dec[j] += levelerr_vec[vec_back]

    # Find minimum error to determine mode peak
    error[0] = err_dec[n - 1]
    error[n] = err_asc[n - 1]
    for i in range(1, n):
        error[i] = err_asc[i - 1] + err_dec[n - i]

    pos_min = 0
    for i in range(0, n):
        if error[i] < error[pos_min]:
            pos_min = i

    # Now do the full up down regression centered on the pos_min
    left_vec = np.arange(0, n)
    right_vec = np.arange(0, n)

    if pos_min != 0:
        mean_vec[0:pos_min] = y[0:pos_min]
        sumwy_vec[0:pos_min] = y[0:pos_min] * w[0:pos_min]
        sumw_vec[0:pos_min] = w[0:pos_min]

        vec_back = 0
        for j in range(1, pos_min):
            while vec_back >= 0 and mean_vec[j] <= mean_vec[vec_back]:
                left_vec[j] = left_vec[vec_back]
                sumwy_vec[j] += sumwy_vec[vec_back]
                sumw_vec[j] += sumw_vec[vec_back]
                mean_vec[j] = sumwy_vec[j] / sumw_vec[j]
                vec_back -= 1
            vec_back += 1
            left_vec[vec_back] = left_vec[j]
            right_vec[vec_back] = right_vec[j]
            sumwy_vec[vec_back] = sumwy_vec[j]
            sumw_vec[vec_back] = sumw_vec[j]
            mean_vec[vec_back] = mean_vec[j]

        for k in range(0, vec_back+1):
            for l in range(left_vec[k], right_vec[k]+1):
                y_new[l] = mean_vec[k]

    if pos_min != n:
        mean_vec[0:n-pos_min] = y_reverse[0:n-pos_min]
        left_vec[0:n-pos_min] = np.arange(0, n-pos_min)
        right_vec[0:n-pos_min] = np.arange(0, n-pos_min)
        sumwy_vec[0:n-pos_min] = y_reverse[0:n-pos_min] * w_reverse[0:n-pos_min]
        sumw_vec[0:n-pos_min] = w_reverse[0:n-pos_min]

        vec_back = 0
        for j in range(1, n - pos_min):
            while vec_back >= 0 and mean_vec[j] <= mean_vec[vec_back]:
                left_vec[j] = left_vec[vec_back]
                sumwy_vec[j] += sumwy_vec[vec_back]
                sumw_vec[j] += sumw_vec[vec_back]
                mean_vec[j] = sumwy_vec[j] / sumw_vec[j]
                vec_back -= 1
            vec_back += 1
            left_vec[vec_back] = left_vec[j]
            right_vec[vec_back] = right_vec[j]
            sumwy_vec[vec_back] = sumwy_vec[j]
            sumw_vec[vec_back] = sumw_vec[j]
            mean_vec[vec_back] = mean_vec[j]

        for k in range(0, vec_back+1):
            for l in range(left_vec[k], right_vec[k]+1):
                y_new[n - l - 1] = mean_vec[k]

    mode_peak = pos_min - 1 # pos_min is sliceable so subtract one for actual index

    return y_new, mode_peak
