import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy

from util import plotBandSpec

if len(sys.argv) < 2:
    sys.exit("Filename was not introduced")

if len(sys.argv) < 4:
    sys.exit("Start time and end time must introduced. The format is hh:mm:ss")

# set input file (which day to work on and which channel)
filename = sys.argv[1]
stream = obspy.read(filename)

trace = stream[0]

arrTimes = [3600,60,1]

try:
    # start = map(int,sys.argv[2].split(':'))
    # startInSec = sum(a*b for a,b in zip(arrTimes, start))

    startInSec = 1000

    # end = map(int,sys.argv[3].split(':'))
    # endInSec = sum(a*b for a,b in zip(arrTimes,end))

    endInSec = 2000
    print("Hello")

    auxTrace = trace.copy()
    cutData = auxTrace.trim(startInSec,endInSec)
    plotBandSpec(cutData)
except:
    sys.exit("Times may not have been inserted or the format is not correct.")
# Make a plot of the whole day of data
#trace.plot(type='dailyplot', outfile="dailyplot")

# set hour of the day (1-23) to work with
# hr = 1.0

# tzero = trace.stats.starttime

# for i in range(0,71,1):
#     auxTrace = trace.copy()
#     cutData = auxTrace.trim(tzero+(hr+i)*1200,tzero+((hr+i+1)*1200))
#     plotBandSpec(cutData)


