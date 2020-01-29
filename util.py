import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.signal.freqattributes import bandwidth


# Generates and saves a bandpass-filtered data and its spectrogram
# There are two modes: save and plot. "save" mode is default and will save the 
# figure whereas "plot" mode will plot the figure in running time.

def plotBandSpec(trace, mode='save',low=24.99, high=0.001):
    #copy the data

    #demean, detrend, bandpass filter
    trace.detrend('demean')
    trace.detrend('linear')
    trace.filter('bandpass', freqmin=high, freqmax=low, corners=2, zerophase=True)

    traceCopy = trace.copy()

    # Plot the filtered data and spectrogram
    t = np.arange(0, trace.stats.npts / trace.stats.sampling_rate, trace.stats.delta)
    _, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(t, trace.data, 'k')
    ax1.set_ylabel('Bandpassed Data')
    traceCopy.spectrogram(log=True, axes=ax2)
    ax2.set_ylabel('Spectrogram')
    plt.xlabel('Time [s]')
    plt.suptitle(traceCopy.stats.starttime)

    if mode=='plot':
        plt.show()
    elif mode=='save':
        plt.savefig(str(traceCopy.stats.starttime)+".png")
    else:
        print("Unknown value of parameter 'mode'")