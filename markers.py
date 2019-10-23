from pyrocko import util
from pyrocko.gui import marker as pm
from pyrocko.example import get_example_data



def lookPattern(trace, markers):
    kind = -1
    for marker in markers:
        if(trace.stats.starttime >= marker.tmin):
            kind = marker.kind
        else:
            break
    
    return kind
