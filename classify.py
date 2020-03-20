import sys
import numpy as np
import matplotlib.pyplot as plt
import obspy
#from obspy.core import read
#from obspy.signal import freqattributes
from obspy.signal.freqattributes import bandwidth
#from obspy.signal.filter import bandpass

from pyrocko import util
from pyrocko.gui import marker as pm

from markers import lookPattern


###Command line arguments
# 1: number of the day
# 2: number of the marker


if len(sys.argv) < 2:
    sys.exit("Day number and markers number were not introduced.")

# set input file (which day to work on and which channel)
day = sys.argv[1]
dayCMD = formatDay(day)
filename = dict[int(sys.argv[2])] + dayCMD

stream = obspy.read(filename)
markers = pm.load_markers("../Markers/"+dayCMD+".mk")

trace = stream[0]

interval = 30 #seconds for window width
overlap = 15

tzero = trace.stats.starttime

traces = []

maxTime = 24*60*60-interval
#maxTime = 50 #for testing

#fileToSave = open(, 'w')

lastColumn = []

for i in range(0, maxTime,overlap):
    auxTrace = trace.copy()
    cutData = auxTrace.trim(tzero+i,tzero+i+interval)
    addClass = np.append(lastColumn, lookPattern(cutData, markers))
    #print(np.append(cutData.data, lookPattern(cutData, markers))) #If you wint to print activate this




fileToSave.close
