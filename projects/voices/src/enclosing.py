import numpy as np
from scipy.io import wavfile
from ssqueezepy import ssq_cwt, issq_cwt
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.io.wavfile import write
import sys
from sklearn.neighbors import NearestCentroid
import soundfile as sf

if len(sys.argv) < 2:
    print('\n Usage -> python voice-svm <wav file path>\n')
    sys.exit(1)

cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list, fig_width):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(fig_width, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list

def straighten(filename, plot_state=True, inv_state=True):
    '''
    filename: Direction of .wav file.
    plot_state: Demonstration of peaks in before and after being straight.
    inv_state: Hear .wav file after being straightened up.
    
    NOTE_: 
    - Order of differentiation for difftype='numeric' (default=4)
    x, y = x/4
    - Tx: np.ndarray [nf x n]
        Synchrosqueezed CWT of `x`. (rows=~frequencies, cols=timeshifts)
        (nf = len(ssq_freqs); n = len(x))
        `nf = na` by default, where `na = len(scales)`.
    - Wx: np.ndarray [na x n]
        Continuous Wavelet Transform of `x`, L1-normed (see `cwt`).
    nv: int / None
            Number of voices (wavelets per octave). Suggested >= 16.
    '''
    ################################### Read raw data ###################################
    sample_rate, x = wavfile.read(filename) 
    per_milisec = sample_rate/1000 # number of samples per milisecond
    print(f'Sample rate: {per_milisec} samples/milisec  \nSampling frequency: {sample_rate} Hz \nShape of x: {x.shape} \nTotal time: {x.shape[0]/sample_rate} seconds')
    
    nv_scale = sample_rate/197
    Twx, Wx, *_ = ssq_cwt(x) # https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/ssqueezepy/_ssq_cwt.py
    # widths = np.linspace(1, 15, sample_rate//10)
    # widths = np.arange(1, int(per_milisec))
    # Twx = signal.cwt(x, signal.ricker, widths, np.complex128)

    # Compute a spectrogram with consecutive Fourier transforms.:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    f, t, Sxx = signal.spectrogram(x, sample_rate)  

    base = np.abs(Twx)
    peak = base.copy()
    peak_copy = np.zeros_like(peak)
    # height = (np.max(peak) -  np.min(x))/4 # the least height of peaks
    height = 100
    print(f'Threshold height of peaks: {height}')
    ################################### Extract peaks ###################################
    # for each frequency keep only the peaks and transform them to 1, the rest 0 
    for i, row in enumerate(peak):
        peaks, _ = find_peaks(row, height=height)
        mask = np.ones_like(row, bool)
        mask[peaks] = False
        row[mask] = 0
        row[peaks] = 1

    y, x = (peak > 0).nonzero()
    print('Twx: ', Twx.shape)
    dots = np.ones(peak[peak>0].size)
    
    if plot_state:
        fig, axs = plt.subplots(2)
        # plotting the original
        axs[0].pcolormesh(t*1000, f, Sxx, shading='viridis')
        axs[0].scatter(x/per_milisec, y, c=dots, s=1, cmap='Set1')
        axs[0].set_xlim(0, np.max(t*1000))
        axs[0].set_ylim(0, f[7])
        wav_file = filename.split('/')[-1]
        fig.suptitle(wav_file)

    ################################### Creating Standard data ###################################
    unique, counts = np.unique(y, return_counts=True)

    ################################### Grid algorithm ###################################
    dis = 50  # Distance between points
    x_list = np.array(range(min(x), max(x), dis))
    y_list = np.array(range(min(y), max(y), dis))
    x_1, y_1 = np.meshgrid(x_list, y_list)   # a grid is created by standard straighten

    x_train = np.array([x_1[:, i] for i in range(x_1.shape[1])]).reshape(-1, )
    y_train = np.array([y_1[:, i] for i in range(y_1.shape[1])]).reshape(-1, )

    y_train = np.concatenate((y_train.reshape(-1, 1), x_train.reshape(-1, 1)), axis=-1)

    
    ################################### Train with ML ###################################
    # This standard data ((y_train, x_train), x_train) is trained with a Machine Learning (ML) model(NearestCentroid). 
    # After that, using the machine learning
    # model predict original data ((y, x), y)
    clf = NearestCentroid()
    clf.fit(y_train, x_train)
    x = clf.predict(np.concatenate((y.reshape(-1, 1), x.reshape(-1, 1)), axis=-1))
    peak_copy[y, x] = 1
    dots = np.ones(peak[peak>0].size)

    if plot_state:
        axs[1].pcolormesh(t*1000, f, Sxx, shading='nearest', cmap='viridis')
        axs[1].scatter(x/per_milisec, y, c=dots, s=1, cmap='Set1') # plotting the straightened data (x, y)
        axs[1].set_xlim(0, np.max(t*1000))
        axs[1].set_ylim(0, f[7])
        for ax in axs.flat:
            ax.set(xlabel='Time (miliseconds)', ylabel='Frequency (kHz)')
        fig_width, fig_height = fig.get_size_inches()  # Size of image
 
        plt.savefig('straighten.png')  # save image of demonstration the straightened data (x, y)
        plot_color_gradients('Lowest to highest density of frequency: ', ['viridis'], fig_width)
        plt.show()
    
    ################################### Invert back to original signal ###################################
    if inv_state == True:
        Twx_copy = Twx.copy() 
        for idx, Twx_e in enumerate(Twx):
            # Note: peak_old_len's always bigger than peak_new_len
            peak_old_len = len(Twx_e[peak[idx]>0])         # number of old peaks on each row
            peak_new_len = len(Twx_e[peak_copy[idx]>0])    # number of new peaks on each row
            
            if peak_new_len > 0:
                power = float(peak_old_len//peak_new_len)  # the ratio of the old peaks's length on the new peaks's length
            
            peak_old_value = Twx_e[peak[idx]>0]
            if len(peak_old_value)>0:
                Twx_copy[idx][peak_copy[idx]>0] = peak_old_value[:peak_new_len]  # convert old peaks to new peaks
            
            if peak_new_len > 0:
                Twx_copy[idx][peak_copy[idx]<1] /= (power**power)  # all of the low frequency (except old peaks) were set to the value of power^power
        x_2 = issq_cwt(Twx_copy)
        # sf.write('inverse.wav', x_2.astype(np.int32), sample_rate, 'PCM_24')
        write("inverse.wav", sample_rate, x_2.astype(np.int16))


if __name__ == '__main__':
    # python (filename.py) (filesound.wav): python /home/ubuntu-pc/Enrico_boss/voices/projects/voices/src/enclosing.py /home/ubuntu-pc/Enrico_boss/voices/wav/apple_and_lemmon.wav
    filename = sys.argv[1]
    straighten(filename, plot_state=True, inv_state=True)