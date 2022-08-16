from pathlib import Path

import h5py
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, localize_sfm
from hloc.utils.io import get_matches, get_keypoints, find_pair, read_image
from hloc.utils.parsers import parse_retrieval
from hloc.utils.viz import plot_images, plot_matches, save_plot
from tqdm import tqdm

query_images = Path(
    '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_images/')
db_descriptors = \
    '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db/global-feats-openibl.h5'
db_local_descriptors = \
    '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db/feats-superpoint-n4096-rmax1600.h5'

query_outputs = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query')
query_image_retrieval_pairs = query_outputs / 'pairs.txt'

retrieval_conf = extract_features.confs['openibl'] #'dir' = ResNet - deep image retrieval
feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']

# qi_paths = []
# qi_paths += list(Path(query_images).glob("*.jpg"))

# get the K closest db images for each query image
query_descriptors = extract_features.main(retrieval_conf, query_images, query_outputs)
pairs_from_retrieval.main(query_descriptors, query_image_retrieval_pairs, num_matched=3, db_descriptors=Path(db_descriptors))

# match the query images to their already retrieved closest neighbours
query_features = extract_features.main(feature_conf, query_images, query_outputs)
matches = match_features.main(matcher_conf, query_image_retrieval_pairs, feature_conf['output'], query_outputs, features_ref = Path(db_local_descriptors))

# vis matches
ret = parse_retrieval(query_image_retrieval_pairs)
pairs = [(q, r) for q, rs in ret.items() for r in rs]
for pair in tqdm(pairs):
    matches_kp_idx_img_1_idx_2 , _ = get_matches(matches, pair[0], pair[1])
    kp_q = get_keypoints(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query/feats-superpoint-n4096-rmax1600.h5'), pair[0])
    kp_db = get_keypoints(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db/feats-superpoint-n4096-rmax1600.h5'), pair[1])

    q_image = read_image(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_images') / pair[0])
    db_image = read_image(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/mesh_data/') / pair[1])
    plot_images([q_image, db_image], dpi=100)
    plot_matches(kp_q[matches_kp_idx_img_1_idx_2[0:25,0]], kp_db[matches_kp_idx_img_1_idx_2[0:25,1]], lw=2.5, color=(0,1,0), a=.1)
    save_plot('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/vis_debug/' + pair[0]+"_"+pair[1])