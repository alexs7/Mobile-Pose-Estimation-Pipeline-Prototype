## ExMaps: Long-Term Localization in Dynamic Scenes using Exponential Decay

### This repo contains the code for the above paper accepted in WACV2021.


The basic commands to run for generating the results are:

    python3 get_visibility_matrix.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/ 
    python3 get_points_3D_mean_descs.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/ 
    python3 main.py /home/user/fullpipeline/colmap_data/CMU_data/slice3/

The directory `/home/user/fullpipeline/colmap_data/CMU_data/slice3/` will be different in your case. 

`get_visibility_matrix.py applies` the exponential decay.

One you run this code for all the CMU slices (or retail shop) then you will want to run `results_analyzer.py`. 

Notes: 

 1. This repo is still under construction. For any questions please
        contact ar2056(at)bath.ac.uk.   
 2. The data will have to be added for the
        code to run. Once you do add it it is easy to replace it with yours.
