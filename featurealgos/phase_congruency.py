import numpy as np
import cv2


def construct_log_gabor_filters(rows, cols, n_scales, n_orientations,
                                min_wavelength, mult, sigma_on_f, d_theta_on_sigma):
    u_range = np.fft.fftshift(np.fft.fftfreq(cols))
    v_range = np.fft.fftshift(np.fft.fftfreq(rows))

    x, y = np.meshgrid(u_range, v_range)

    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-y, x)

    radius[rows//2, cols//2] = 1

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    filters = []

    wavelengths = [min_wavelength * (mult ** i) for i in range(n_scales)]
    center_freqs = [1.0 / w for w in wavelengths]

    theta_interval = np.pi / n_orientations
    theta_sigma = theta_interval / d_theta_on_sigma

    for o in range(n_orientations):
        angl = o * theta_interval

        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp((-dtheta**2) / (2 * theta_sigma**2))

        scale_filters = []
        for s in range(n_scales):
            fo = center_freqs[s]
            log_gabor = np.exp((-(np.log(radius/fo))**2) /
                               (2 * np.log(sigma_on_f)**2))

            log_gabor[rows//2, cols//2] = 0

            filt = log_gabor * spread
            scale_filters.append(filt)

        filters.append(scale_filters)

    return np.array(filters), wavelengths


def phase_congruency(image, n_scales=4, n_orientations=6, min_wavelength=3, mult=2.0,
                     sigma_on_f=0.55, k=2.0, cut_off=0.4, g=10, epsilon=0.001):
    # 1. Pre-processing
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float64)
    rows, cols = img.shape

    image_fft = np.fft.fft2(img)

    # 2. Construct Filters
    d_theta_on_sigma = 1.2
    filters, _ = construct_log_gabor_filters(rows, cols, n_scales, n_orientations,
                                             min_wavelength, mult, sigma_on_f, d_theta_on_sigma)

    total_energy_result = np.zeros((rows, cols))
    total_amplitude_sum = np.zeros((rows, cols))

    # 3. Process Orientations
    for o in range(n_orientations):
        sum_real = np.zeros((rows, cols))
        sum_imag = np.zeros((rows, cols))
        sum_amp_o = np.zeros((rows, cols))

        max_amp_o = np.zeros((rows, cols))

        scale_amps = []
        scale_reals = []
        scale_imags = []

        filter_power_0 = 0
        median_response = 0

        # 4. Process Scales
        for s in range(n_scales):
            filt = filters[o, s]

            filt_shifted = np.fft.ifftshift(filt)

            response_fft = image_fft * filt_shifted
            response_spatial = np.fft.ifft2(response_fft)

            real = np.real(response_spatial)
            imag = np.imag(response_spatial)
            amp = np.sqrt(real**2 + imag**2)

            scale_amps.append(amp)
            scale_reals.append(real)
            scale_imags.append(imag)

            sum_real += real
            sum_imag += imag
            sum_amp_o += amp

            max_amp_o = np.maximum(max_amp_o, amp)

            if s == 0:
                median_response = np.median(amp**2)
                filter_power_0 = np.sum(filt**2)

        # 5. Noise Compensation
        if filter_power_0 == 0:
            expected_energy_sq_total = 0
        else:
            expected_energy_sq_0 = -median_response / np.log(0.5)
            sum_filter_powers = np.sum(
                [np.sum(filters[o, s]**2) for s in range(n_scales)])
            power_ratio = sum_filter_powers / filter_power_0
            expected_energy_sq_total = expected_energy_sq_0 * power_ratio

        sigma_g_sq = expected_energy_sq_total / 2.0
        mu_r = np.sqrt(sigma_g_sq * np.pi / 2.0)
        sigma_r_sq = (4 - np.pi) / 2.0 * sigma_g_sq
        sigma_r = np.sqrt(sigma_r_sq)

        T = mu_r + k * sigma_r

        # 6. Calculate Phase Deviation and Weighting
        energy_vector_mag = np.sqrt(sum_real**2 + sum_imag**2) + epsilon
        mean_phase_unit_real = sum_real / energy_vector_mag
        mean_phase_unit_imag = sum_imag / energy_vector_mag

        phase_dev_sum = np.zeros((rows, cols))

        for s in range(n_scales):
            dot_prod = scale_reals[s] * mean_phase_unit_real + \
                scale_imags[s] * mean_phase_unit_imag
            cross_prod = scale_reals[s] * mean_phase_unit_imag - \
                scale_imags[s] * mean_phase_unit_real

            phase_dev_sum += (dot_prod - np.abs(cross_prod))

        energy_o = np.maximum(phase_dev_sum - T, 0)

        width = (sum_amp_o / (max_amp_o + epsilon)) / n_scales

        weight_o = 1.0 / (1.0 + np.exp(g * (cut_off - width)))

        total_energy_result += (weight_o * energy_o)

        total_amplitude_sum += sum_amp_o

    # 7. Final Calculation
    PC = total_energy_result / (total_amplitude_sum + epsilon)

    PC = np.clip(PC, 0, 1)

    return PC


def compute_sift_dominant_orientation(image, x, y, size=10):
    rows, cols = image.shape
    radius = int(np.round(size))

    min_r, max_r = max(0, int(y) - radius), min(rows, int(y) + radius + 1)
    min_c, max_c = max(0, int(x) - radius), min(cols, int(x) + radius + 1)

    patch = image[min_r:max_r, min_c:max_c]

    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return 0.0

    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)

    magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    p_rows, p_cols = patch.shape
    sigma = radius / 1.5
    gy, gx = np.meshgrid(np.arange(-p_rows//2, p_rows//2),
                         np.arange(-p_cols//2, p_cols//2), indexing='ij')

    gy = gy[:p_rows, :p_cols]
    gx = gx[:p_rows, :p_cols]

    gaussian_weight = np.exp(-(gx**2 + gy**2) / (2 * sigma**2))

    weighted_magnitude = magnitude * gaussian_weight

    n_bins = 36
    hist = np.zeros(n_bins)

    bins = (angle / (360.0 / n_bins)).astype(int) % n_bins

    np.add.at(hist, bins.ravel(), weighted_magnitude.ravel())

    peak_bin = np.argmax(hist)

    dominant_angle = (peak_bin * (360.0 / n_bins)) + (360.0 / n_bins / 2)

    return dominant_angle
