import colmap

def register_image(db_path, query_images_dir, image_list_file, existing_model_dir, new_model_dir, queryExtraction=False):
    colmap.feature_extractor(db_path, query_images_dir, image_list_file, query=queryExtraction)
    colmap.vocab_tree_matcher(db_path, image_list_file)
    colmap.image_registrator(db_path, existing_model_dir, new_model_dir)
