# 13/10/2020 - This is used only for simple recontructions - cmu_* is used instead for the paper benchmarks
import sys
import os
import colmap

db_path = sys.argv[1]
images_dir = sys.argv[2]
sparse_model_dir = sys.argv[3]

colmap.feature_extractor(db_path, images_dir)
colmap.vocab_tree_matcher(db_path)
colmap.mapper(db_path, images_dir, sparse_model_dir)
