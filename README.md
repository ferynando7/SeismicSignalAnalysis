################
To copy the file structure of YACHAY folder please type this command:
    find . -type d > dirs.txt

To create the file structure in the folder you wish:
    xargs mkdir -p < dirs.txt
################

##########################################
Script: openSnuffler.py


    This script opens snuffler in a particular day and with the stations and channels specified. The parameters are:
        - day = from 1 to 119
        - tracks: 
            11: Station BMAS, channel BHE.D
            12: Station BMAS, channel BHN.D
            13: Station BMAS, channel BHE.Z
            21: Station BPAT, channel BHE.D
            22: Station BPAT, channel BHN.D
            23: Station BPAT, channel BHE.Z
            31: Station BRUN, channel BHE.D
            32: Station BRUN, channel BHN.D
            33: Station BRUN, channel BHE.Z
            41: Station BULB, channel BHE.D
            42: Station BULB, channel BHN.D
            43: Station BULB, channel BHE.Z}
           
        - marker file: the marker file name must be the last parameter. It is optional

    The format of the command is the following:

        python openSnuffler.py <day> <track_1> ... <track_n> <markerFileName>

    Example:
        python openSnuffler.py 10 11 22 31 markers.pf

        This means that snuffler will be opened in the 10th day with the following stations and channels:

            - BMAS in channel BHE.D
            - BPAT in channel BHN.D
            - BRUM in channel BHE.D
            - The marker file is markers.pf

#######################################