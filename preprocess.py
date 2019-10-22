import sys
import numpy as np
from obspy import read


if(len(sys.argv) == 1):
    sys.exit("Filename was not specified")


filename = sys.argv[1]
interval = 6 #five seconds for each window 
data = read(filename)

trace = data[0]

### Parameters of the trace ###
metadata = trace.stats
sr = metadata.sampling_rate
npts = metadata.npts
###

widthWindow = sr*interval
totalSeconds = (int) (npts // sr) #total time of the signals

start = metadata.starttime
end = metadata.endtime



for i in range(0,totalSeconds, (int) (interval // 2)):
    a = trace.copy()
    print(a.trim(start + i, start + i + interval))

# print(trace)
# print(metadata)

# print(tr.stats)
# data = tr.data()

# print(tr.data.mean())
# print(tr.data.std())

# print(data)

# st.plot()
