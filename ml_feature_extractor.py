import colmap
import sys

db_path = sys.argv[1] #"/home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_db.db"
query_images_dir = sys.argv[2] #"/home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/test_images/"
image_list_file = sys.argv[3] #"/home/alex/fullpipeline/colmap_data/Coop_data/slice1/ML_data/images_list.txt"

colmap.feature_extractor(db_path, query_images_dir, image_list_file, query=True)
