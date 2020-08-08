import glob
import os
import sys
import random
import subprocess

# This will move images so be careful!

source_folder = sys.argv[1]
dest_folder = sys.argv[2]
random_images_no = sys.argv[3]

os.chdir(source_folder)
for folder in glob.glob("*"):
    images = []
    for image in glob.glob(folder+"/*.jpg"):
        basename = image.split("/")[1]
        images.append(basename)

    random_images = random.sample(images,20)

    for random_image in random_images:
        subprocess.run(["mv", source_folder+folder+"/"+random_image, dest_folder])
