import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.signal.util import next_pow_2
from obspy.signal.freqattributes import central_frequency_unwindowed, spectrum, welch
from obspy.signal.filter import bandpass
from util import dict, formatDay




##########################################
def getMetrics(trace):
    data = trace.data
    mean = data.mean()
    median = np.median(data) 
    stdv = data.std()
    maximum = np.amax(data)
    trace.taper(type='hamming',max_percentage=0.05, max_length=5)
    data = trace.data
    repFreq = central_frequency_unwindowed(data,df)
    filtered = bandpass(data, 0.01, 1.5625, df)
    sumEnergy = np.sum(welch(filtered, np.hamming(len(data)), next_pow_2(len(data))))
    return [mean, median, stdv, maximum, repFreq, sumEnergy]
##########################################

##########################################

def processFile(filename):
    stream = obspy.read(filename)

    trace = stream[0]
    trace.detrend('demean')
    trace.detrend('linear')

    tzero = trace.stats.starttime


    maxTime = 24*60*60-interval

    metrics = []
    for i in range(0, maxTime, overlap):
        auxTrace = trace.copy()
        cutData = auxTrace.trim(tzero+i,tzero+i+interval)
        metrics.append(getMetrics(cutData))


    for i in range(len(metrics)):
        if i == 0:
            metrics[i].append(0)
        else:
            if metrics[i-1][5] < metrics[i][5]:
                metrics[i].append(1)
            elif metrics[i-1][5] > metrics[i][5]:
                metrics[i].append(-1)
            else:
                metrics[i].append(0)

    for i in range(len(metrics)):
        if i == len(metrics)-1:
            metrics[i].append(0)
        else:
            if metrics[i][5] < metrics[i+1][5]:
                metrics[i].append(1)
            elif metrics[i][5] > metrics[i-1][5]:
                metrics[i].append(-1)
            else:
                metrics[i].append(0)
    return metrics
##########################################

if len(sys.argv) < 2:
    sys.exit("Day was not introduced")


# set input file (which day to work on and which channel)
day = formatDay(int(sys.argv[1]))

df = 50 #sampling rate
interval = 30 #seconds for window width
overlap = 15

header = 'MEAN_E,MEDIAN_E,STDV_E,MAXIMUM_E,REP_FREQ_E,SUM_ENERGY_E,ENERGY_PREV_E,ENERGY_NEXT_E,MEAN_N,MEDIAN_N,STDV_N,MAXIMUM_N,REP_FREQ_N,SUM_ENERGY_N,ENERGY_PREV_N,ENERGY_NEXT_N,MEAN_Z,MEDIAN_Z,STDV_Z,MAXIMUM_Z,REP_FREQ_Z,SUM_ENERGY_Z,ENERGY_PREV_Z,ENERGY_NEXT_Z'

filenames = filenames = [prefix+day for prefix in dict.values()]

##BMAS
station1 = processFile(filenames[0])
station2 = processFile(filenames[1])
station3 = processFile(filenames[2])

bmas = np.hstack((station1,station2,station3))

outputFile = "../Windowed/BMAS." + day
fileToSave = open(outputFile+'.csv', 'w')
np.savetxt(fileToSave, bmas, delimiter=',', header=header)
fileToSave.close


##BPAT
station1 = processFile(filenames[3])
station2 = processFile(filenames[4])
station3 = processFile(filenames[5])

bpat = np.hstack((station1,station2,station3))

outputFile = "../Windowed/BPAT." + day
fileToSave = open(outputFile+'.csv', 'w')
np.savetxt(fileToSave, bpat, delimiter=',', header=header)
fileToSave.close


##BRUN
station1 = processFile(filenames[6])
station2 = processFile(filenames[7])
station3 = processFile(filenames[8])

brun = np.hstack((station1,station2,station3))

outputFile = "../Windowed/BRUN." + day
fileToSave = open(outputFile+'.csv', 'w')
np.savetxt(fileToSave, brun, delimiter=',', header=header)
fileToSave.close

##BULB
station1 = processFile(filenames[9])
station2 = processFile(filenames[10])
station3 = processFile(filenames[11])

bulb = np.hstack((station1,station2,station3))

outputFile = "../Windowed/BULB." + day
fileToSave = open(outputFile+'.csv', 'w')
np.savetxt(fileToSave, bulb, delimiter=',', header=header)
fileToSave.close

#metrics = processFile(filename)

