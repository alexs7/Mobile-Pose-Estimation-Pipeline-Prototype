rm colmap_data/data/images/* && rm -rf colmap_data/data/model/0 && rm colmap_data/data/database.db &&
	python3 sparse_reconstuctor.py colmap_data/data/database.db colmap_data/data/images colmap_data/data/model

