import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.signal.freqattributes import bandwidth, central_frequency_unwindowed
from obspy.signal import util
from scipy import fftpack, signal
from operator import itemgetter



dict = {
    11: "../YACHAY/BMAS/BHE.D/EC.BMAS..BHE.D.2010.",
    12: "../YACHAY/BMAS/BHN.D/EC.BMAS..BHN.D.2010.",
    13: "../YACHAY/BMAS/BHZ.D/EC.BMAS..BHZ.D.2010.",
    21: "../YACHAY/BPAT/BHE.D/EC.BPAT..BHE.D.2010.",
    22: "../YACHAY/BPAT/BHN.D/EC.BPAT..BHN.D.2010.",
    23: "../YACHAY/BPAT/BHZ.D/EC.BPAT..BHZ.D.2010.",
    31: "../YACHAY/BRUN/BHE.D/EC.BRUN..BHE.D.2010.",
    32: "../YACHAY/BRUN/BHN.D/EC.BRUN..BHN.D.2010.",
    33: "../YACHAY/BRUN/BHZ.D/EC.BRUN..BHZ.D.2010.",
    41: "../YACHAY/BULB/BHE.D/EC.BULB..BHE.D.2010.",
    42: "../YACHAY/BULB/BHN.D/EC.BULB..BHN.D.2010.",
    43: "../YACHAY/BULB/BHZ.D/EC.BULB..BHZ.D.2010."
}


def formatDay(day):
    dayCMD = ""
    if day < 10:
        dayCMD = "00"+str(day)
    elif day <100:
        dayCMD = "0"+str(day)
    else:
        dayCMD = str(day)
    return dayCMD

# Generates and saves a bandpass-filtered data and its spectrogram
# There are two modes: save and plot. "save" mode is default and will save the 
# figure whereas "plot" mode will plot the figure in running time.

def plotBandSpec(trace, mode='save',low=24.99, high=0.001):
    #copy the data
    df = 50
    #demean, detrend, bandpass filter
    trace.detrend('demean')
    trace.detrend('linear')
    trace.filter('bandpass', freqmin=high, freqmax=low, corners=2, zerophase=True)

    traceCopy = trace.copy()
    data = traceCopy.data
    repFreq = central_frequency_unwindowed(data,df)
    bw = bandwidth(data, df)
    print("Central frequency: " + str(repFreq) + "\nBandwidth: " + str(bw))



    # # Plot the filtered data and spectrogram
    # t = np.arange(0, trace.stats.npts / trace.stats.sampling_rate, trace.stats.delta)
    # _, (ax1, ax2) = plt.subplots(2, sharex=True)
    # ax1.plot(t, trace.data, 'k')
    # ax1.set_ylabel('Bandpassed Data')
    # traceCopy.spectrogram(log=True, axes=ax2)
    # ax2.set_ylabel('Spectrogram')
    # plt.xlabel('Time [s]')
    # plt.suptitle(traceCopy.stats.starttime)

    # if mode=='plot':
    #     plt.show()
    # elif mode=='save':
    #     plt.savefig(str(traceCopy.stats.starttime)+".png")
    # else:
    #     print("Unknown value of parameter 'mode'")



def bandwidth(data, fs):
    """
    Bandwidth of a signal.

    Computes the bandwidth of the given data which can be windowed or not.
    The bandwidth corresponds to the level where the power of the spectrum is
    half its maximum value. It is determined as the level of 1/sqrt(2) times
    the maximum Fourier amplitude.

    If data are windowed the bandwidth of each window is returned.

    :type data: :class:`~numpy.ndarray`
    :param data: Data to make envelope of.
    :param fs: Sampling frequency in Hz.
    :param smoothie: Factor for smoothing the result.
    :param fk: Coefficients for calculating time derivatives
        (calculated via central difference).
    :return: **bandwidth[, dbwithd]** - Bandwidth, Time derivative of
        predominant period (windowed only).
    """
    new_dtype = np.float32 if data.dtype.itemsize == 4 else np.float64
    data = np.require(data, dtype=new_dtype)

    nfft = util.next_pow_2(data.shape[0])
    freqaxis = np.linspace(0, fs, nfft + 1)
    bwith = np.zeros(data.shape[0])
    
    minfc = abs(data - max(abs(data * (1 / np.sqrt(2)))))
    [mdist_ind, _mindist] = min(enumerate(minfc), key=itemgetter(1))
    bwith = freqaxis[mdist_ind]
    return bwith

