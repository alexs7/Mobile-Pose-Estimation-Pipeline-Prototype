class RANSACParameters(object):
    # PROSAC sorting values for matches indices
    lowes_distance_inverse_ratio_index = 0
    higher_visibility_score_index = 1
    heatmap_val_index = 2
    reliability_score_ratio_index = 3
    lowes_ratio_reliability_score_val_ratio_index = 4
    higher_neighbour_score_index = 5
    reliability_score_index = 6
    heatmap_val_ratio_index = 7
    higher_neighbour_val_index = 8
    lowes_ratio_heatmap_val_ratio_index = 9
    lowes_ratio_by_higher_reliability_score_index = 10
    lowes_ratio_by_higher_heatmap_val_index = 11

    prosac_value_titles = {lowes_distance_inverse_ratio_index: "inverse_lowes_ratio",
                           heatmap_val_index: "heatmap_value",
                           reliability_score_index: "reliability_score",
                           reliability_score_ratio_index: "reliability_score_ratio",
                           lowes_ratio_reliability_score_val_ratio_index: "lowes_by_reliability_score_ratio",
                           lowes_ratio_heatmap_val_ratio_index: "lowes_by_heatmap_value_ratio",
                           higher_neighbour_score_index: "reliability_higher_neighbour_score",
                           heatmap_val_ratio_index: "heatmap_value_ratio",
                           higher_neighbour_val_index: "reliability_higher_neighbour_heatmap_value",
                           higher_visibility_score_index: "higher_neighbour_visibility_score",
                           lowes_ratio_by_higher_reliability_score_index: "lowes_by_higher_neighbour_reliability_score",
                           lowes_ratio_by_higher_heatmap_val_index: "lowes_by_higher_neighbour_heatmap_value"}

    use_ransac_dist_heatmap_val = -1
    use_ransac_dist_reliability_score = -2
    use_ransac_dist_visibility_score = -3

    ransac_base = "ransac_base"
    prosac_base = "prosac_base"
    ransac_live = "ransac_live"
    ransac_dist_heatmap_val = "ransac_dist_heatmap_val"
    ransac_dist_reliability_score = "ransac_dist_reliability_score"
    ransac_dist_visibility_score = "ransac_dist_visibility_score"

    ransac_prosac_iterations = 3000  # I used 1000 for dev
    ransac_prosac_error_threshold = 5.0  # 8.0 default, 5 from the Benchamrking 6DoF long term localization