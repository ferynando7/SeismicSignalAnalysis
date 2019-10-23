import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy
#from obspy.core import read
#from obspy.signal import freqattributes
from obspy.signal.freqattributes import bandwidth
#from obspy.signal.filter import bandpass

from util import plotBandSpec

if len(sys.argv) < 2:
    sys.exit("Filename was not introduced")

# set input file (which day to work on and which channel)
filename = sys.argv[1]
stream = obspy.read(filename)

trace = stream[0]

# Make a plot of the whole day of data
trace.plot(type='dailyplot', outfile="dailyplot")

# set hour of the day (1-23) to work with
hr = 1.0

tzero = trace.stats.starttime

for i in range(0,71,1):
    auxTrace = trace.copy()
    cutData = auxTrace.trim(tzero+(hr+i)*1200,tzero+((hr+i+1)*1200))
    plotBandSpec(cutData)


