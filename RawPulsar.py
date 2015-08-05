__author__ = 'monika'

# Algorithm based on the 'Generation of a Coherent Dispersed Pulse
# by Simulation' paper by Deshpande and Ellingson

import math
import struct
import numpy as np
import matplotlib.pyplot as plt

alpha = 2.410e-16  # for DM in [pc cm ^{ -3}] , time in [s] , and frequency in [Hz]

class RawPulsar:

    def __init__(self,
                 freq_sampling=7.5e+6,
                 fft_size=1024,
                 freq_centre=38.0e+6,
                 sim_time=1.0,
                 ):

        """ Initialise the parameters"""
        self.freq_sampling = freq_sampling  # [ samples per second ]
        self.t_samp = 1/self.freq_sampling  # Time domain sample period
        self.fft_size = fft_size  # size of FFT
        self.f_samp = self.freq_sampling/self.fft_size  # FFT bin width
        self.fft_period = self.fft_size * self.t_samp  # FFT period
        self.freq_centre = freq_centre  # [Hz] center frequency of passband (as received )
        self.sim_time = sim_time  # duration of simulation
        self.fft_blocks = math.ceil(self.sim_time/self.fft_period)  # number of FFT input blocks to process
        self.n_samp = self.fft_blocks*self.fft_size  # number of samples
        self.raw_data = np.array([])

    def _initialise_fft_bins(self):
        """
        FFT bin center frequencies (at baseband) [Hz]
        :return:
        """
        self.fft_bin = np.arange(-self.freq_sampling/2, self.freq_sampling/2, self.f_samp)

    def _initialise_start_freq(self):
        """
        start frequency (center freq of pulse at start of simulation) [Hz]
        :return:
        """
        self._initialise_fft_bins()
        self.fch1 = self.freq_centre + max(self.fft_bin)/2

    def _initialise_start_freq_delay(self, dm):
        """
        reference delay for start freq (f1) i.e. the shortest delay at the highest freq [s]
        :param dm: dispersion measure
        :return:
        """
        self._initialise_start_freq()
        self.dch1 = dm/(alpha*(self.fch1**2))

    def _initialise_delay_bins(self, dm):
        """

        :param dm: dispersion measure
        :return:
        """
        self._initialise_start_freq_delay(dm)
        fc = self.freq_centre
        df = self.f_samp
        f2 = fc + self.fft_bin - df/2
        t1 = self.dch1
        tbin = (dm/(alpha*(f2**2))) - t1  # [s] delay relative to t1 for lowest frequency in each bin
        fbin = self.fft_bin
        lfft = self.fft_size
        self.relative_delay = np.append(tbin, (dm/(alpha*((fc+fbin[lfft-1]+df/2)**2)))-t1)  # to simplify algorithm

    def initialise_data(self, dm):
        """

        :param dm: dispersion measure
        :return:
        """
        self._initialise_delay_bins(dm)
        self.raw_data = np.zeros(self.n_samp, dtype=np.complex64)

    def write_pulse_to_bin(self, duty_cycle=0.05, t=0):
        """

        :param duty_cycle: the fraction of the pulse period in which the pulse is actually on
        :param t: time
        :return: vector with delayed pulse
        """
        b = 0  # count bins
        beta = 1/math.log(2)  # constant
        tbin = self.relative_delay
        X = [0] * self.fft_size
        for f in range(len(self.fft_bin)):  # loop over FFT bins
            t2 = tbin[b+1]  # delay for max frequency in this bin
            X[b] = math.exp(-(t - t2)**2/(beta * duty_cycle ** 2))  # Gaussian shaped pulse
            b += 1
        return X

    def fftshift(self, x):
        """
        FFT shift into screwy FFT order
        :param x:
        :return:
        """
        return np.fft.fftshift(x)

    def ifft(self, x):
        """
        Inverse FFT gives time domain
        :param x:
        :return:
        """
        return np.fft.ifft(x)

    def add_pulsar(self,
                   period_s=.1,
                   duty_cycle=0.05,
                   # dm=5.,
                   # snr = 1.
                   ):
        """

        :param period_s: the time between two pulses
        :param duty_cycle: the fraction of the pulse period in which the pulse is actually on
        :return:
        """
        t = 0  # [s] initialize sim time
        k = 0
        lfft = self.fft_size
        while k < self.fft_blocks:
            k += 1
            X = self.write_pulse_to_bin(duty_cycle, t)
            t += self.fft_period  # update time
            X = self.fftshift(X)
            x = self.ifft(X)*lfft
            self.raw_data[(k-1)*lfft:k*lfft] = x

    def write_file(self, file_name='test'):
        """

        :param file_name:
        :return:
        """
        xs = self.raw_data
        p_amp = 1  # amplitude of the pulse in time domain
        max_xs = max(np.absolute(xs))
        with open(file_name, 'ab') as myfile:
            for l in range(len(xs)):
                xr = int(round(10*(p_amp*np.real(xs[l])/max_xs+np.random.standard_normal())))
                xi = int(round(10*(p_amp*np.imag(xs[l])/max_xs+np.random.standard_normal())))
                mybuffer = struct.pack("bb", xr, xi)
                myfile.write(mybuffer)

    def plot(self, file_name='test'):
        """

        :param file_name:
        :return:
        """
        with open(file_name, 'rb') as fh:
            loaded_array = np.frombuffer(fh.read(), dtype=np.int8)

        xr = loaded_array[0:len(loaded_array)-1:2]
        xi = loaded_array[1:len(loaded_array):2]
        cX = xr + xi*1j
        fc = self.freq_centre
        fbin = self.fft_bin
        lfft = self.fft_size
        p1 = 1
        power_array = np.zeros((lfft, len(cX)/lfft))
        for p in range(len(cX)/lfft):
            power = np.abs(np.fft.fft(cX[(p1-1)*lfft:p1*lfft], lfft))**2
            p1 += 1
            print 'p1 = %d' % p1
            power_array[:, p] = power
        power_array = np.fliplr(power_array)

        plt.imshow(power_array, extent=[0, self.sim_time, fc-max(fbin), fc+max(fbin)], aspect='auto')
        plt.show()


if __name__ == '__main__':
    rP = RawPulsar(freq_sampling=7.5e+6, fft_size=1024, freq_centre=38.0e+6, sim_time=1)
    print 'Time domain sample period    dt = %.2e' % rP.t_samp
    print 'FFT bin width                df = %.2e' % rP.f_samp
    print 'FFT period                   dt2 = %.2e' % rP.fft_period
    rP.initialise_data(dm=5.)
    print 'Start frequency              f1 = %.2e' % rP.fch1
    print 'FFT blocks to process        kmax = %d' % rP.fft_blocks
    rP.add_pulsar(period_s=.1, duty_cycle=0.05)
    rP.write_file('test_2')
    rP.plot('test_2')

