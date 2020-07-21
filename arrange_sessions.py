import glob
import os
import sys
import numpy as np

# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live
# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt

def gen_query_txt(dir, save_sessions = False):
    session_nums = []
    images = []
    for folder in glob.glob(dir+"/session_*"):
        i=0
        for file in glob.glob(folder+"/*.jpg"):
            image_name = file.split('images/')[1]
            images.append(image_name)
            i+=1
        session_nums.append(i)

    with open(dir+'/../query_name.txt', 'w') as f:
        for image in images:
            f.write("%s\n" % image)

    print("session_nums = " + str(session_nums) + " - Dont forget to add the base ones!")

    if(save_sessions) : np.savetxt(dir+"/../session_lengths.txt", session_nums)
