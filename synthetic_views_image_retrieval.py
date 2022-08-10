# This file will be used to build an image retrieval system for retrieving top k similar images
# compared to a query image
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils.parsers import parse_retrieval

db_images = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/synth_images')
query_images = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_image_single')

db_outputs = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db')
db_image_retrieval_pairs = db_outputs / 'pairs-netvlad.txt'
query_outputs = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query')
query_image_retrieval_pairs = query_outputs / 'pairs-netvlad.txt'

# sfm_dir = outputs / 'sfm_superpoint_max+superglue'

retrieval_conf = extract_features.confs['openibl'] #'dir' = ResNet - deep image retrieval
# feature_conf = extract_features.confs['superpoint_max']
# matcher_conf = match_features.confs['superglue']

print("Extracting global descriptors images..")

db_descriptors = extract_features.main(retrieval_conf, db_images, db_outputs)
query_descriptors = extract_features.main(retrieval_conf, query_images, query_outputs)

pairs_from_retrieval.main(query_descriptors, query_image_retrieval_pairs, num_matched=3, db_descriptors=db_descriptors)

res = parse_retrieval(query_image_retrieval_pairs)

breakpoint()
