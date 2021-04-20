from image_registrator import register_image
import sys

db_path = sys.argv[1] #"colmap_data/data/database.db"
query_images_dir = sys.argv[2] #"colmap_data/data/current_query_image"
query_images_list_file = sys.argv[3] #"colmap_data/data/query_name.txt"
current_model = sys.argv[4] #"colmap_data/data/model/0"
output_model = sys.argv[5] #"colmap_data/data/new_model"

# the database here that has to be used is the on from the sfm.
# you could another db (an empty) but it makes no sense - no time to investigate
# for ML purposes run this command

# python3 register_query_image.py /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/database.db /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_images/2020-06-22 /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/model /home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/new_model

register_image(db_path,
               query_images_dir,
               query_images_list_file,
               current_model,
               output_model,
               queryExtraction=True)


