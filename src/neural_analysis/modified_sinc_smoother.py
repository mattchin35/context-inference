"""
From Schimid et al, ACS Measurement Science 2022
"""

import numpy as np


def cube(x):
    return x * x * x


def fit_weighted(x_data, y_data, weights):
    """Weighted linear fit of the data."""
    sum_weights = np.sum(weights)
    sum_x = np.sum(x_data * weights)
    sum_y = np.sum(y_data * weights)
    sum_x2 = np.sum(x_data * x_data * weights)
    sum_xy = np.sum(x_data * y_data * weights)

    var_x2 = sum_x2 * sum_weights - sum_x * sum_x
    if var_x2 == 0:
        slope = 0
    else:
        slope = (sum_xy * sum_weights - sum_x * sum_y) / var_x2

    offset = (sum_y - slope * sum_x) / sum_weights
    return offset, slope


def extend_data(data, m, fit_weights):
    """Extends the data by weighted linear extrapolation, for smoothing to the ends."""
    dat_length = len(data)
    fit_length = len(fit_weights)

    fit_x = np.arange(1, fit_length + 1)

    # Left edge extrapolation
    fit_y_left = data[:fit_length]
    offset_left, slope_left = fit_weighted(fit_x, fit_y_left, fit_weights)

    # Pre-allocate extended array
    ext_data = np.zeros(dat_length + 2 * m)

    # Populate left extrapolation
    ext_data[:m] = offset_left + np.arange(-m + 1, 1) * slope_left

    # Populate center with original data
    ext_data[m: dat_length + m] = data

    # Right edge extrapolation
    fit_y_right = data[dat_length - fit_length: dat_length][::-1]
    offset_right, slope_right = fit_weighted(fit_x, fit_y_right, fit_weights)

    # Populate right extrapolation
    ext_data[dat_length + m: dat_length + 2 * m] = offset_right + np.arange(0, -m, -1) * slope_right

    return ext_data


def edge_weights(deg, m):
    """Hann-square weights for linear fit at the edges, for MS smoothing."""
    beta = 0.70 + 0.14 * np.exp(-0.6 * (deg - 4))
    fit_length_d = ((m + 1) * beta) / (1.5 + 0.5 * deg)
    fit_length = int(np.floor(fit_length_d))

    w = np.zeros(fit_length + 1)
    for i in range(1, fit_length + 2):
        cosine = np.cos(np.pi / 2 * (i - 1) / fit_length_d)
        w[i - 1] = cosine * cosine
    return w


def edge_weights1(deg, m):
    """Hann-square weights for linear fit at the edges, for MS1 smoothing."""
    beta = 0.65 + 0.35 * np.exp(-0.55 * (deg - 4))
    fit_length_d = ((m + 1) * beta) / (1 + 0.5 * deg)
    fit_length = int(np.floor(fit_length_d))

    w = np.zeros(fit_length + 1)
    for i in range(1, fit_length + 2):
        cosine = np.cos(np.pi / 2 * (i - 1) / fit_length_d)
        w[i - 1] = cosine * cosine
    return w


def window_ms(x, alpha):
    """Gaussian-like window function for the MS and MS1 kernels."""
    w = np.exp(-alpha * x * x) + np.exp(-alpha * (x + 2) * (x + 2)) + np.exp(-alpha * (x - 2) * (x - 2)) \
        - (2 * np.exp(-alpha) + np.exp(-9 * alpha))
    return w


def corr_coeffs_ms(deg):
    """Correction coefficients for a flat passband of the MS kernel."""
    if deg == 2:
        return np.array([])
    elif deg == 4:
        return np.array([])
    elif deg == 6:
        return np.array([[0.001717576, 0.02437382, 1.64375]])
    elif deg == 8:
        return np.array([[0.0043993373, 0.088211164, 2.359375],
                         [0.006146815, 0.024715371, 3.6359375]])
    elif deg == 10:
        return np.array([[0.0011840032, 0.04219344, 2.746875],
                         [0.0036718843, 0.12780383, 2.7703125]])
    else:
        raise ValueError("Invalid deg")


def corr_coeffs_ms1(deg):
    """Correction coefficients for a flat passband of the MS1 kernel."""
    if deg == 2:
        return np.array([])
    elif deg == 4:
        return np.array([[0.021944195, 0.050284006, 0.765625]])
    elif deg == 6:
        return np.array([[0.001897730, 0.00847681, 1.2625],
                         [0.023064667, 0.13047926, 1.2265625]])
    elif deg == 8:
        return np.array([[0.006590300, 0.05792946, 1.915625],
                         [0.002323448, 0.01029885, 2.2726562],
                         [0.021046653, 0.16646601, 1.98125]])
    elif deg == 10:
        return np.array([[9.749618E-4, 0.00207429, 3.74375],
                         [0.008975366, 0.09902466, 2.707812],
                         [0.002419541, 0.01006486, 3.296875],
                         [0.019185117, 0.18953617, 2.784961]])
    else:
        raise ValueError("Invalid deg")


def kernel_ms(deg, m):
    """Calculates the MS convolution kernel."""
    coeffs = corr_coeffs_ms(deg)
    kappa = []

    if len(coeffs) > 0:
        for abcd in coeffs:
            kappa.append(abcd[0] + abcd[1] / cube(abcd[2] - m))

    if (deg / 2) % 2 == 1:  # degree 6, 10
        nu_minus_2 = -1
    else:
        nu_minus_2 = 0

    kernel = np.zeros(2 * m + 1)
    kernel[m] = window_ms(0, 4)  # center element

    for i in range(1, m + 1):
        x = i / (m + 1)
        w = window_ms(x, 4)
        a = np.sin((0.5 * deg + 2) * np.pi * x) / ((0.5 * deg + 2) * np.pi * x)
        for j in range(len(kappa)):
            a = a + kappa[j] * x * np.sin((2 * (j + 1) + nu_minus_2) * np.pi * x)
        a = a * w
        kernel[m - i] = a
        kernel[m + i] = a

    norm = np.sum(kernel)
    kernel = kernel / norm
    return kernel


def kernel_ms1(deg, m):
    """Calculates the MS1 convolution kernel."""
    coeffs = corr_coeffs_ms1(deg)
    kappa = []

    if len(coeffs) > 0:
        for abcd in coeffs:
            kappa.append(abcd[0] + abcd[1] / cube(abcd[2] - m))

    kernel = np.zeros(2 * m + 1)
    kernel[m] = window_ms(0, 2)  # center element

    for i in range(1, m + 1):
        x = i / (m + 1)
        w = window_ms(x, 2)
        a = np.sin((0.5 * deg + 1) * np.pi * x) / ((0.5 * deg + 1) * np.pi * x)
        for j in range(len(kappa)):
            a = a + kappa[j] * x * np.sin((j + 1) * np.pi * x)
        a = a * w
        kernel[m - i] = a
        kernel[m + i] = a

    norm = np.sum(kernel)
    kernel = kernel / norm
    return kernel


def smooth_ms(data, deg, m):
    """
    Smooths data using the Modified Sinc (MS) kernel.
    data: 1D numpy array
    deg: degree (2, 4, 6, 8, 10)
    m: half-width of the kernel
    """
    if len(data) < 2:
        raise ValueError("Less than two data points")

    kernel = kernel_ms(deg, m)
    fit_weights = edge_weights(deg, m)
    ext_data = extend_data(data, m, fit_weights)

    # "same" mode in numpy matches Octave's conv(..., "same")
    smoothed_ext_data = np.convolve(ext_data, kernel, mode='same')

    # Strip off the extrapolated edges
    smoothed_data = smoothed_ext_data[m: -m]
    return smoothed_data


def smooth_ms1(data, deg, m):
    """
    Smooths data using the shorter MS1 kernel.
    """
    if len(data) < 2:
        raise ValueError("Less than two data points")

    kernel = kernel_ms1(deg, m)
    fit_weights = edge_weights1(deg, m)
    ext_data = extend_data(data, m, fit_weights)

    smoothed_ext_data = np.convolve(ext_data, kernel, mode='same')
    smoothed_data = smoothed_ext_data[m: -m]
    return smoothed_data


# ==========================================
# TEST CODE (Same as the Octave script)
# ==========================================
if __name__ == "__main__":
    deg = 6
    m = 7
    data = np.array([0, 1, -2, 3, -4, 5, -6, 7, -8, 9, 10, 6, 3, 1, 0])

    print("smoothMS of test data:")
    out = smooth_ms(data, deg, m)

    # Print the output formatted to match the Octave output exactly
    formatted_out = " ".join([f"{val:10.8f}" for val in out])
    print(formatted_out)