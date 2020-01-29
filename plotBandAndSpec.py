import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy

from util import plotBandSpec

if len(sys.argv) < 2:
    sys.exit("Filename was not introduced")

if len(sys.argv) < 4:
    sys.exit("Start time and end time must introduced. The format is hh:mm:ss")

if len(sys.argv) < 5:
    sys.exit("Highpass and/or lowpass must be specified")

# set input file (which day to work on and which channel)
filename = sys.argv[1]
stream = obspy.read(filename)

trace = stream[0]

arrTimes = [3600,60,1]

try:
    start = map(int,sys.argv[2].split(':'))
    startInSec = sum(a*b for a,b in zip(arrTimes, start))

    end = map(int,sys.argv[3].split(':'))
    endInSec = sum(a*b for a,b in zip(arrTimes,end))

   
except:
    sys.exit("Times may not have been inserted or the format is not correct.")



tzero = trace.stats.starttime

auxTrace = trace.copy()
cutData = auxTrace.trim(tzero + startInSec, tzero + endInSec)

[high,low] = sys.argv[4].split('-')
if high == 'l':
    plotBandSpec(cutData, mode = 'plot', low = float(low))
elif high == 'h':
    plotBandSpec(cutData, mode = 'plot', high = float(low))
else:
    plotBandSpec(cutData, mode = 'plot', low=float(low), high= float(high))


# Make a plot of the whole day of data
#trace.plot(type='dailyplot', outfile="dailyplot")

# set hour of the day (1-23) to work with
# hr = 1.0


# for i in range(0,71,1):
#     auxTrace = trace.copy()
#     cutData = auxTrace.trim(tzero+(hr+i)*1200,tzero+((hr+i+1)*1200))
#     plotBandSpec(cutData)


