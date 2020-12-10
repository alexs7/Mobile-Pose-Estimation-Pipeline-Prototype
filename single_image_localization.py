import colmap
from feature_matcher_single_image import feature_matcher_wrapper

db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/database.db"
query_images_dir = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image"
image_list_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt"

# Step 1: Feature Extractor
colmap.feature_extractor(db_path, query_images_dir, image_list_file)

# Step 2: Feature Matching
feature_matcher_wrapper()

# Step 3: Solver

# Step 4: Appl