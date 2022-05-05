import glob
from image_registrator import register_image
import sys
import os

base_path = sys.argv[1]
db_path = os.path.join(base_path,"database.db")
query_images_dir = os.path.join(base_path,"scale_data")
query_images_list_file = os.path.join(base_path,"query_name.txt")
current_model = os.path.join(base_path,"model/0/")
output_model = os.path.join(base_path,"new_model/")

image_files = [f for f in glob.glob(query_images_dir + '/'+ '*.jpg')]

try: # remove file
    os.remove(query_images_list_file)
except OSError:
    pass

with open(query_images_list_file, 'a') as f:
    for image_file in image_files:
        name = image_file.split("/")[-1]
        f.write(name + "\n")

register_image(db_path,
               query_images_dir,
               query_images_list_file,
               current_model,
               output_model)
