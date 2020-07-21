from image_registrator import register_image
import sys

db_path = sys.argv[1] #"colmap_data/data/database.db"
query_images_dir = sys.argv[2] #"colmap_data/data/current_query_image"
query_images_list_file = sys.argv[3] #"colmap_data/data/query_name.txt"
current_model = sys.argv[4] #"colmap_data/data/model/0"
output_model = sys.argv[5] #"colmap_data/data/new_model"

register_image(db_path,
               query_images_dir,
               query_images_list_file,
               current_model,
               output_model)
