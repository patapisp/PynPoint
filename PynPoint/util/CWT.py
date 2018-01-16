import numpy as np
import continous as wave
from scipy.special import gamma, hermite
from scipy.signal import medfilt
from statsmodels.robust import mad
from numba import jit
import copy

import matplotlib.pyplot as plt

# --- Wavelet analysis Capsule ---------
# TODO: Documentation


@jit(cache=True)
def _fast_zeros(soft,
                spectrum,
                uthresh):
        if soft:
            for i in range(0, spectrum.shape[0], 1):
                for j in range(0, spectrum.shape[1], 1):
                    tmp_value = spectrum[i, j].real
                    if abs(spectrum[i, j]) > uthresh:
                        spectrum[i, j] = np.sign(tmp_value) * (abs(tmp_value) - uthresh)
                    else:
                        spectrum[i, j] = 0
        else:
            for i in range(0, spectrum.shape[0], 1):
                for j in range(0, spectrum.shape[1], 1):
                    if abs(spectrum[i, j]) < uthresh:
                        spectrum[i, j] = 0

        return spectrum


@jit(cache=True)
def _fast_zeros_planet_save(spectrum,
                            uthresh,
                            uplanet):

    for i in range(0, spectrum.shape[0], 1):
        for j in range(0, spectrum.shape[1], 1):
            if abs(spectrum[i, j]) < uthresh * uplanet[i]:
                spectrum[i, j] = 0

    return spectrum


class WaveletAnalysisCapsule:

    def __init__(self,
                 signal_in,
                 wavelet_in='dog',
                 order=2,
                 padding="none",
                 frequency_resolution=0.1):

        # save input data
        self._m_supported_wavelets = ['dog', 'morlet']

        # check supported wavelets
        if not (wavelet_in in self._m_supported_wavelets):
            raise ValueError('Wavelet ' + str(wavelet_in) + ' is not supported')

        if wavelet_in == 'dog':
            self._m_C_reconstructions = {2: 3.5987,
                                          4: 2.4014,
                                          6: 1.9212,
                                          8: 1.6467,
                                          12: 1.3307,
                                          16: 1.1464,
                                          20: 1.0222,
                                          30: 0.8312,
                                          40: 0.7183,
                                          60: 0.5853}
        elif wavelet_in == 'morlet':
            self._m_C_reconstructions = {5: 0.9484,
                                          6: 0.7784,
                                          7: 0.6616,
                                          8: 0.5758,
                                          10: 0.4579,
                                          12: 0.3804,
                                          14: 0.3254,
                                          16: 0.2844,
                                          20: 0.2272}
        self._m_wavelet = wavelet_in

        if padding not in ["none", "zero", "mirror"]:
            raise ValueError("Padding can only be none, zero or mirror")

        self._m_data = signal_in - np.ones(len(signal_in)) * np.mean(signal_in)
        self._m_padding = padding
        self._pad_signal()
        self._m_data_size = len(self._m_data)
        self._m_data_mean = np.mean(signal_in)

        if order not in self._m_C_reconstructions:
            raise ValueError('Wavelet ' + str(wavelet_in) + ' does not support order ' + str(order) +
                             ". \n Only orders: " + str(sorted(self._m_C_reconstructions.keys())).strip('[]') +
                             " are supported")
        self._m_order = order
        self._m_C_final_reconstruction = self._m_C_reconstructions[order]

        # create scales for wavelet transform
        self._m_scales = wave.autoscales(N = self._m_data_size,
                                          dt=1,
                                          dj=frequency_resolution,
                                          wf=wavelet_in,
                                          p=order)

        self._m_number_of_scales = len(self._m_scales)
        self._m_frequency_resolution = frequency_resolution

        self._m_spectrum = None
        return

    # --- functions for reconstruction value
    @staticmethod
    def _morlet_function(omega0,
                          x):
        return np.pi**(-0.25) * np.exp(1j * omega0 * x) * np.exp(-x**2/2.0)

    @staticmethod
    def _dog_function(order,
                       x):
        pHpoly = hermite(order)[int(x / np.power(2, 0.5))]
        herm = pHpoly / (np.power(2, float(order) / 2))
        return ((-1)**(order+1)) / np.sqrt(gamma(order + 0.5)) * herm

    def _pad_signal(self):
        padding_length = int(len(self._m_data) * 0.5)
        if self._m_padding == "none":
            return

        elif self._m_padding == "zero":
            new_data = np.append(self._m_data, np.zeros(padding_length, dtype=np.float64))
            self._m_data = np.append(np.zeros(padding_length, dtype=np.float64), new_data)

        else:
            # Mirror Padding
            left_half_signal = self._m_data[:padding_length]
            right_half_signal = self._m_data[padding_length:]
            new_data = np.append(self._m_data, right_half_signal[::-1])
            self._m_data = np.append(left_half_signal[::-1], new_data)

    def _compute_reconstruction_factor(self):
        dj = self._m_frequency_resolution
        wavelet = self._m_wavelet
        order = self._m_order

        if wavelet == 'morlet':
            zero_function = self._morlet_function(order, 0)
        else:
            zero_function = self._dog_function(order, 0)

        c_delta = self._m_C_final_reconstruction

        reconstruction_factor = dj/(c_delta * zero_function)
        return reconstruction_factor.real

    def compute_cwt(self):
        self._m_spectrum = wave.cwt(self._m_data,
                                     dt=1,
                                     scales=self._m_scales,
                                     wf=self._m_wavelet,
                                     p=self._m_order)

    def update_signal(self):
        self._m_data = wave.icwt(self._m_spectrum,
                                  dt=1,
                                  scales=self._m_scales,
                                  wf=self._m_wavelet,
                                  p=self._m_order)
        reconstruction_factor = self._compute_reconstruction_factor()
        self._m_data *= reconstruction_factor

    def _transform_period(self,
                           period):

        tmp_y = wave.fourier_from_scales(self._m_scales, self._m_wavelet,self._m_order)

        def transformation(x):
            return np.log2(x + 1) * tmp_y[-1] / np.log2(tmp_y[-1] + 1)

        cutoff_scaled = transformation(period)

        scale_new = tmp_y[-1] - tmp_y[0]
        scale_old = self._m_spectrum.shape[0]

        factor = scale_old / scale_new
        cutoff_scaled *= factor

        return cutoff_scaled

    def denoise_spectrum_universal_threshold(self,
                                             threshold=1.0,
                                             soft=False):

        if not self._m_padding == "none":
            noise_length_4 = len(self._m_data)/4
            noise_spectrum = self._m_spectrum[0, noise_length_4: (noise_length_4*3)].real
        else:
            noise_spectrum = self._m_spectrum[0, :].real

        sigma = mad(noise_spectrum)
        uthresh = sigma*np.sqrt(2.0*np.log(len(noise_spectrum))) * threshold

        self._m_spectrum = _fast_zeros(soft,
                                        self._m_spectrum,
                                        uthresh)

    def median_filter(self):
        self._m_data = medfilt(self._m_data, 19)

    def get_signal(self):

        tmp_data = self._m_data + np.ones(len(self._m_data))*self._m_data_mean
        if self._m_padding == "none":
            return tmp_data
        else:
            return tmp_data[len(self._m_data)/4: 3*len(self._m_data)/4]

    # ----- plotting functions --------

    def _plot_or_save_spectrum(self):
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.subplot(1, 1, 1)

        tmp_y = wave.fourier_from_scales(self._m_scales, self._m_wavelet,self._m_order)
        tmp_x = np.arange(0, self._m_data_size + 1, 1)

        scaled_spec = copy.deepcopy(self._m_spectrum.real)

        for i in range(len(scaled_spec)):
            scaled_spec[i] /= np.sqrt(self._m_scales[i])

        print scaled_spec.shape
        plt.imshow(abs(scaled_spec),
                   aspect='auto',
                   extent=[tmp_x[0],
                           tmp_x[-1],
                           tmp_y[0],
                           tmp_y[-1]],
                   cmap=plt.get_cmap("gist_ncar"),
                   origin='lower')

        # TODO if for no padding
        # COI first part (only for DOG) with padding

        inner_frequency = 2.*np.pi/np.sqrt(self._m_order + 0.5)
        coi = np.append(np.zeros(len(tmp_x)/4),
                        tmp_x[0:len(tmp_x) / 4])
        coi = np.append(coi,
                        tmp_x[0:len(tmp_x) / 4][::-1])
        coi = np.append(coi,
                        np.zeros(len(tmp_x) / 4))

        plt.plot(np.arange(0, len(coi), 1.0),
                 inner_frequency * coi / np.sqrt(2),
                 color="white")

        plt.ylim([tmp_y[0],
                  tmp_y[-1]])

        plt.fill_between(np.arange(0, len(coi) , 1.0),
                         inner_frequency * coi / np.sqrt(2),
                         np.ones(len(coi)) * tmp_y[-1],
                         facecolor="none",
                         edgecolor='white',
                         alpha=0.4,
                         hatch="x")

        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: "%.3f" % (np.exp(y))))
        plt.yscale('log', basey=2)
        plt.ylabel("Period in [s]")
        plt.xlabel("Time in [s]")
        plt.title("Spectrum computed with CWT using '" + str(self._m_wavelet) +
                  "' wavelet of order " + str(self._m_order))

    def plot_spectrum(self):
        self._plot_or_save_spectrum()
        plt.show()

    def save_spectrum(self,
                      location):
        self._plot_or_save_spectrum()
        plt.savefig(location)
        plt.close()

    def _plot_or_save_signal(self):
        plt.close()
        plt.plot(self._m_data)
        plt.title("Signal")
        plt.ylabel("Value of the function")
        plt.xlim([0, self._m_data_size])
        plt.xlabel("Time in [s]")

    def plot_signal(self):
        self._plot_or_save_signal()
        plt.show()

    def save_signal(self,
                    location):
        self._plot_or_save_signal()
        plt.savefig(location)

    # ---------------------------------
