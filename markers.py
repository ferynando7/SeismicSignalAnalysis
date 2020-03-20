from pyrocko import util
from pyrocko.gui import marker as pm
from pyrocko.example import get_example_data



def lookPattern(trace, markers):
    label = -1
    i = 0
    for marker in markers:
        if(trace.stats.starttime >= marker.tmin and trace.stats.starttime <= marker.tmax):
            label = marker.kind
        elif(trace.stats.endtime >= marker.tmin and trace.stats.endtime <= marker.tmax):
            label = marker.kind
    return label
