import os
import sys


dict = {
    11: "../YACHAY/BMAS/BHE.D/EC.BMAS..BHE.D.2010.",
    12: "../YACHAY/BMAS/BHN.D/EC.BMAS..BHN.D.2010.",
    13: "../YACHAY/BMAS/BHZ.D/EC.BMAS..BHZ.D.2010.",
    21: "../YACHAY/BPAT/BHE.D/EC.BPAT..BHE.D.2010.",
    22: "../YACHAY/BPAT/BHN.D/EC.BPAT..BHN.D.2010.",
    23: "../YACHAY/BPAT/BHZ.D/EC.BPAT..BHZ.D.2010.",
    31: "../YACHAY/BRUN/BHE.D/EC.BRUN..BHE.D.2010.",
    32: "../YACHAY/BRUN/BHN.D/EC.BRUN..BHN.D.2010.",
    33: "../YACHAY/BRUN/BHZ.D/EC.BRUN..BHZ.D.2010.",
    41: "../YACHAY/BULB/BHE.D/EC.BULB..BHE.D.2010.",
    42: "../YACHAY/BULB/BHN.D/EC.BULB..BHN.D.2010.",
    43: "../YACHAY/BULB/BHZ.D/EC.BULB..BHZ.D.2010."
}

if len(sys.argv) < 2:
    sys.exit("Day was not specified")
elif not sys.argv[1].isnumeric or int(sys.argv[1]) < 1 or int(sys.argv[1]) >119:
    sys.exit("Day must be numeric, greater than 0 and less or equal than 119")

day = int(sys.argv[1])
dayCMD = ""
if day < 10:
    dayCMD = "00"+str(day)
elif day <100:
    dayCMD = "0"+str(day)
else:
    dayCMD = str(day)

command = "snuffler "

for i in range(2,len(sys.argv)):
    try: #in case sys.argv[i] is a number
        track = dict[int(sys.argv[i])] + dayCMD + " "
        command = command + track
    except:
        if sys.argv[i] == sys.argv[-1]:
            command = command + "--markers " + sys.argv[i]
            break
        else:
            sys.exit("Options for track are defined in README.md. Markers file name should be the last parameter.")
os.system(command)