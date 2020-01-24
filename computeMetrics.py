import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.signal.util import next_pow_2
from obspy.signal.freqattributes import central_frequency_unwindowed, spectrum

from pyrocko import util
from pyrocko.gui import marker as pm

from markers import lookPattern

if len(sys.argv) < 2:
    sys.exit("Input filename was not introduced")


##########################################
def getMetrics(trace):
    data = trace.data
    mean = data.mean()
    median = np.median(data) 
    stdv = data.std()
    maximum = np.amax(data)
    repFreq = central_frequency_unwindowed(data,1)
#    spec = spectrum(data,np.hamming(len(data)), next_pow_2(len(data)))
    return [mean, median, stdv, maximum, repFreq]
##########################################

# set input file (which day to work on and which channel)
filename = sys.argv[1]
stream = obspy.read(filename)

trace = stream[0]

interval = 30 #seconds for window width
overlap = 15

tzero = trace.stats.starttime

traces = []

maxTime = 24*60*60-interval

metrics = []
for i in range(0, maxTime, overlap):
    auxTrace = trace.copy()
    cutData = auxTrace.trim(tzero+i,tzero+i+interval)
    metrics.append(getMetrics(cutData))

outputFile = filename.replace('YACHAY', 'Windowed', 1)
header = 'MEAN,MEDIAN,STDV,MAXIMUM,REP_FREQ,SUM_ENERGY,ENERGY_PREV,ENERGY_NEXT,TYPE'
fileToSave = open(outputFile+'.csv', 'w')
np.savetxt(fileToSave, metrics, delimiter=',', header=header)
fileToSave.close
