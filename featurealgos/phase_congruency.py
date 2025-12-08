import numpy as np
import cv2


def construct_log_gabor_filters(rows, cols, n_scales, n_orientations,
                                min_wavelength, mult, sigma_on_f, d_theta_on_sigma):
    dtype = np.float64

    u_range = np.fft.fftshift(np.fft.fftfreq(cols)).astype(dtype)
    v_range = np.fft.fftshift(np.fft.fftfreq(rows)).astype(dtype)

    x, y = np.meshgrid(u_range, v_range)

    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-y, x)

    radius[rows//2, cols//2] = 1.0

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    theta_interval = np.pi / n_orientations
    theta_sigma = theta_interval / d_theta_on_sigma

    angles = (np.arange(n_orientations) * theta_interval).reshape(-1, 1, 1)

    st = sintheta[None, :, :]
    ct = costheta[None, :, :]

    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    ds = st * cos_angles - ct * sin_angles
    dc = ct * cos_angles + st * sin_angles

    dtheta = np.abs(np.arctan2(ds, dc))
    spread = np.exp((-dtheta**2) / (2 * theta_sigma**2))

    wavelengths = min_wavelength * (mult ** np.arange(n_scales, dtype=dtype))
    center_freqs = 1.0 / wavelengths

    fo = center_freqs.reshape(-1, 1, 1)

    log_radius = np.log(radius)[None, :, :]
    log_fo = np.log(fo)

    log_gabor = np.exp((-(log_radius - log_fo)**2) /
                       (2 * np.log(sigma_on_f)**2))

    log_gabor[:, rows//2, cols//2] = 0.0

    filters = spread[:, None, :, :] * log_gabor[None, :, :, :]

    return filters


def phase_congruency(image, n_scales=4, n_orientations=6, min_wavelength=3, mult=2.0,
                     sigma_on_f=0.55, k=2.0, cut_off=0.4, g=10, epsilon=0.001):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = image.astype(np.float64)
    rows, cols = img.shape

    image_fft = np.fft.fft2(img)

    d_theta_on_sigma = 1.2
    filters = construct_log_gabor_filters(rows, cols, n_scales, n_orientations,
                                          min_wavelength, mult, sigma_on_f, d_theta_on_sigma)

    filters_shifted = np.fft.ifftshift(filters, axes=(-2, -1))

    response_fft = image_fft[None, None, :, :] * filters_shifted

    response_spatial = np.fft.ifft2(response_fft)

    real = np.real(response_spatial)
    imag = np.imag(response_spatial)
    amp = np.sqrt(real**2 + imag**2)

    sum_real = np.sum(real, axis=1)
    sum_imag = np.sum(imag, axis=1)
    sum_amp = np.sum(amp, axis=1)

    max_amp = np.max(amp, axis=1)

    median_response = np.median(amp[:, 0]**2, axis=(1, 2))

    filter_power_0 = np.sum(filters[:, 0]**2, axis=(1, 2))

    sum_filter_powers = np.sum(filters**2, axis=(1, 2, 3))

    expected_energy_sq_total = np.zeros_like(median_response)
    valid_power = filter_power_0 > 0

    if np.any(valid_power):
        expected_energy_sq_0 = -median_response[valid_power] / np.log(0.5)
        power_ratio = sum_filter_powers[valid_power] / \
            filter_power_0[valid_power]
        expected_energy_sq_total[valid_power] = expected_energy_sq_0 * power_ratio

    sigma_g_sq = expected_energy_sq_total / 2.0
    sigma_r = np.sqrt((4 - np.pi) / 2.0 * sigma_g_sq)
    mu_r = np.sqrt(sigma_g_sq * np.pi / 2.0)

    T = mu_r + k * sigma_r

    T = T[:, None, None]

    energy_vector_mag = np.sqrt(sum_real**2 + sum_imag**2) + epsilon
    mean_phase_unit_real = sum_real / energy_vector_mag
    mean_phase_unit_imag = sum_imag / energy_vector_mag

    mpu_real = mean_phase_unit_real[:, None, :, :]
    mpu_imag = mean_phase_unit_imag[:, None, :, :]

    dot_prod = real * mpu_real + imag * mpu_imag
    cross_prod = real * mpu_imag - imag * mpu_real

    phase_dev = dot_prod - np.abs(cross_prod)

    phase_dev_sum = np.sum(phase_dev, axis=1)

    energy_o = np.maximum(phase_dev_sum - T, 0)

    width = (sum_amp / (max_amp + epsilon)) / n_scales
    weight_o = 1.0 / (1.0 + np.exp(g * (cut_off - width)))

    weighted_energy_o = weight_o * energy_o

    total_energy_result = np.sum(weighted_energy_o, axis=0)
    total_amplitude_sum = np.sum(sum_amp, axis=0)

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
