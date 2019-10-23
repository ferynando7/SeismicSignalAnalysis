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

if len(sys.argv) < 3:
    if len(sys.argv) < 2:
        sys.exit("Filenames for data and markers were not introduced.")
    else:
        sys.exit("Filename for markers was not introduced")

# set input file (which day to work on and which channel)
filename = sys.argv[1]
stream = obspy.read(filename)
markers = pm.load_markers(sys.argv[2])

trace = stream[0]

interval = 5 #seconds for window width

tzero = trace.stats.starttime

traces = []

#maxTime = 24*60*60-interval
maxTime = 50

fileToSave = open('data.txt', 'w')

for i in range(0, maxTime,interval):
    auxTrace = trace.copy()
    cutData = auxTrace.trim(tzero+i,tzero+i+interval)
    addClass = np.append(cutData.data, lookPattern(cutData, markers))
    np.savetxt(fileToSave, addClass ,newline=" ")
    fileToSave.write("\n")
    #print(np.append(cutData.data, lookPattern(cutData, markers))) #If you wint to print activate this
fileToSave.close
