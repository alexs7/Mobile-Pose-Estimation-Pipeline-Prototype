# This file is to be used to separate files in folders (per weather condition) from the query folder or database. Only to use with CMU-Seasons dataset
# The ones in database folder are used for reconstruction
#  Note: rm session_1/* session_2/* session_3/* session_4/* session_5/* session_6/* session_7/* session_8/* session_9/* -> removed all seperated files if needed
# the folder has to look like this:
# /Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/official_datasets/CMU-Seasons-Extended/slice2/ (whatever number)
# some folders have to be created manually

# run these below in 'base dir'
# rm -rf undistorted/*
# mkdir -p undistorted/base undistorted/session_1/ undistorted/session_2/ undistorted/session_3/ undistorted/session_4/ undistorted/session_5/ undistorted/session_6/ undistorted/session_7/ undistorted/session_8/ undistorted/session_9/

import glob
import os
import sys
from datetime import datetime

import cv2
import numpy as np

Sunny_No_Foliage = [] # 4 Apr 2011
Sunny_Foliage = [] # 1 Sep 2010, 15 Sep 2010, 19 Oct 2010
Cloudy_Foliage = [] # 1 Oct 2010
Overcast_Mixed_Foliage = [] # 28 Oct 2010 (or 26?!)
Low_Sun_Mixed_Foliage = [] # 3 Nov 2010, 12 Nov 2010
Cloudy_Mixed_Foliage = [] # 22 Nov 2010
Low_Sun_No_Foliage_Snow = [] # 21 Dec 2010
Low_Sun_Foliage = [] # 4 Mar 2011
Overcast_Foliage = [] # 28 Jul 2011

i = 0
# example: "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/official_datasets/CMU-Seasons-Extended/slice2/"
# dont forget trailing "/"
base_dir = sys.argv[1]
query_images_folder = "query/"
query_undistorted_images_folder = "undistorted/"
database_images_folder = "database/"
database_undistorted_images_folder = "undistorted/base/"

def undistort(img):
    distortion_params = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
    fx = 868.993378
    fy = 866.063001
    cx = 525.942323
    cy = 420.042529
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    undst = cv2.undistort(img, K, distortion_params, None, K)
    return undst

i=0
os.chdir(base_dir+database_images_folder)
for file in glob.glob("*.jpg"):
    if(file.split('_')[2] == 'c0'):
        i += 1
        img = cv2.imread(file)
        undistorted_img = undistort(img)
        cv2.imwrite("../" + database_undistorted_images_folder + file, undistorted_img)

print("Database Sizes:")
print("Total: " + str(i))

i=0
os.chdir("../"+query_images_folder)
for file in glob.glob("*.jpg"):
    if(file.split('_')[2] == 'c0'):
        i += 1
        timestamp = int(file.split('_')[3].split('us')[0])
        dt = datetime.fromtimestamp(timestamp/1000000)
        day = dt.day
        month = dt.month

        if(day == 4 and month == 4):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite(base_dir + query_undistorted_images_folder + "session_1/" + file, undistorted_img) # session 1 will have no images (they are the dtabase images)
            Sunny_No_Foliage.append(file)
        if((day == 1 and month == 9) or (day == 15 and month == 9) or (day == 19 and month == 10)):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_2/" + file, undistorted_img)
            Sunny_Foliage.append(file)
        if(day == 1 and month == 10):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_3/" + file, undistorted_img)
            Cloudy_Foliage.append(file)
        if(day == 26 and month == 10): #this should be 28/10 but I think they made a mistake it is 26/10
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_4/" + file, undistorted_img)
            Overcast_Mixed_Foliage.append(file)
        if((day == 3 and month == 11) or (day == 12 and month == 11)):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_5/" + file, undistorted_img)
            Low_Sun_Mixed_Foliage.append(file)
        if(day == 22 and month == 11):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_6/" + file, undistorted_img)
            Cloudy_Mixed_Foliage.append(file)
        if(day == 21 and month == 12):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_7/" + file, undistorted_img)
            Low_Sun_No_Foliage_Snow.append(file)
        if(day == 4 and month == 3):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_8/" + file, undistorted_img)
            Low_Sun_Foliage.append(file)
        if (day == 28 and month == 7):
            img = cv2.imread(file)
            undistorted_img = undistort(img)
            cv2.imwrite("../" + query_undistorted_images_folder + "session_9/" + file, undistorted_img)
            Overcast_Foliage.append(file)

print("Query Sizes:")
print("Total: " + str(i))
print("Sunny_No_Foliage: " +str(len(Sunny_No_Foliage)))
print("Sunny_Foliage: " +str(len(Sunny_Foliage)))
print("Cloudy_Foliage: " +str(len(Cloudy_Foliage)))
print("Overcast_Mixed_Foliage: " +str(len(Overcast_Mixed_Foliage)))
print("Low_Sun_Mixed_Foliage: " +str(len(Low_Sun_Mixed_Foliage)))
print("Cloudy_Mixed_Foliage: " +str(len(Cloudy_Mixed_Foliage)))
print("Low_Sun_No_Foliage_Snow: " +str(len(Low_Sun_No_Foliage_Snow)))
print("Low_Sun_Foliage: " +str(len(Low_Sun_Foliage)))
print("Overcast_Foliage: " +str(len(Overcast_Foliage)))
