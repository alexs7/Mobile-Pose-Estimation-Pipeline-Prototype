import colmap

db_path = "colmap_data/data/database.db"
query_images_dir = "colmap_data/data/future_query_image"
image_list_file = "colmap_data/data/query_name.txt"

colmap.feature_extractor(db_path, query_images_dir, image_list_file)
