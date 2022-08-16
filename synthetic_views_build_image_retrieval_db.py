# This file will be used to build an image retrieval db
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils.parsers import parse_retrieval

# will search for (from hloc): 'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
db_images = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/mesh_data')
db_outputs = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db')
retrieval_conf = extract_features.confs['openibl'] #'dir' = ResNet - deep image retrieval

print("Extracting global descriptors for db images..")
extract_features.main(retrieval_conf, db_images, db_outputs) #db_descriptors

print("Extracting local descriptors for db images..")
feature_conf = extract_features.confs['superpoint_max']
extract_features.main(feature_conf, db_images, db_outputs) #db_local_descriptors