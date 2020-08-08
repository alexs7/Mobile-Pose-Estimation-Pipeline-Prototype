# This file is to be used to separate files in folders (per weather condition) from the query folder or database. Only to use with CMU-Seasons dataset
# The ones in database folder are used for reconstruction

import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# From CMU README
# Sunny + No Foliage (reference) | 4 Apr 2011
# Sunny + Foliage | 1 Sep 2010
# Sunny + Foliage | 15 Sep 2010
# Cloudy + Foliage | 1 Oct 2010
# Sunny + Foliage | 19 Oct 2010
# Overcast + Mixed Foliage | 28 Oct 2010
# Low Sun + Mixed Foliage | 3 Nov 2010
# Low Sun + Mixed Foliage | 12 Nov 2010
# Cloudy + Mixed Foliage | 22 Nov 2010
# Low Sun + No Foliage + Snow | 21 Dec 2010
# Low Sun + Foliage | 4 Mar 2011
# Overcast + Foliage | 28 Jul 2011

# dont forget trailing "/"
source_folder = sys.argv[1]
dest_folder = sys.argv[2]

i=0
os.chdir(source_folder)
for file in glob.glob("*.jpg"):
    if(file.split('_')[2] == 'c0'):
        i += 1
        timestamp = int(file.split('_')[3].split('us')[0])
        dt = datetime.fromtimestamp(timestamp/1000000)
        day = dt.day
        month = dt.month

        if(day == 4 and month == 4):
            Path(dest_folder+"session_1").mkdir(exist_ok=True) # This might be an overhead at every if, but it is the quicker way
            subprocess.run(["cp", file, dest_folder+"session_1"])
        if(day == 1 and month == 9):
            Path(dest_folder + "session_2").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_2"])
        if(day == 15 and month == 9):
            Path(dest_folder + "session_3").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_3"])
        if(day == 1 and month == 10):
            Path(dest_folder + "session_4").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_4"])
        if (day == 19 and month == 10):
            Path(dest_folder + "session_5").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_5"])
        if(day == 26 and month == 10): #this should be 28/10 but I think they made a mistake it is 26/10
            Path(dest_folder + "session_6").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_6"])
        if(day == 3 and month == 11):
            Path(dest_folder + "session_7").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_7"])
        if(day == 12 and month == 11):
            Path(dest_folder + "session_8").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_8"])
        if(day == 22 and month == 11):
            Path(dest_folder + "session_9").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_9"])
        if(day == 21 and month == 12):
            Path(dest_folder + "session_10").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_10"])
        if(day == 4 and month == 3):
            Path(dest_folder + "session_11").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_11"])
        if (day == 28 and month == 7):
            Path(dest_folder + "session_12").mkdir(exist_ok=True)
            subprocess.run(["cp", file, dest_folder + "session_12"])
