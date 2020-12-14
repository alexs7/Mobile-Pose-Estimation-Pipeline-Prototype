import numpy as np
import cv2
from cvxpnpl import pnp
from RANSACParameters import RANSACParameters

MAX_RANSAC_ITERS = RANSACParameters.ransac_prosac_iterations
ERROR_THRESHOLD = RANSACParameters.ransac_prosac_error_threshold
# intrinsics matrix
# NOTE: The assumption is that the camera for the gt images will always be 3.
# As colmap keeps adding cameras as you localise.
# This is because of a COLMAP bug that breaks when I set refining focal length to false.
# Remember that at this point I evaluate query images (gt/images) not session images.
# So I have to pick up their intrinsics which they will be different from base and live.
# TODO: in the future they will all be the same.

def model_refit(img_points, obj_points, K):
    # if image_points >= 4 returns 1 pose otherwise shit hits the fan
    poses = pnp(pts_2d=img_points, pts_3d=obj_points, K=K)
    R, t = poses[0]
    Rt = np.r_[(np.c_[R, t]), [np.array([0, 0, 0, 1])]]
    return Rt

def model_fit(img_points, obj_points, flag_val, K):
    distCoeffs = np.zeros((5, 1))
    # calculate pose
    # this is required for SOLVEPNP_P3P - works with others too
    img_points = np.ascontiguousarray(img_points[:, :2]).reshape((img_points.shape[0], 1, 2))
    # https://answers.opencv.org/question/64315/solvepnp-object-to-camera-pose/
    # Note: there might be a bug with solvePnP on OSX - could try doing this over HTTP on Linux ?
    _, rvec, tvec = cv2.solvePnP(obj_points.astype(np.float32), img_points.astype(np.float32), K, distCoeffs, flags=flag_val)

    # some safeguards
    if(rvec is None):
        rvec = np.zeros([3,1])
    if(tvec is None):
        tvec = np.zeros([3,1])

    rotm = cv2.Rodrigues(rvec)[0]
    Rt = np.r_[(np.c_[rotm, tvec]), [np.array([0, 0, 0, 1])]]
    return Rt

def model_evaluate(matches_for_image, Rt, threshold, K):
    # NOTE: can also do this: points2D, _ = cv2.projectPoints(obj_point[:,0:3].transpose(), rvec, tvec, K, np.zeros((5, 1)))
    obj_point = matches_for_image[:, 2:5]
    img_point_gt = matches_for_image[:, 0:2] #this is wrong
    obj_point = np.hstack((obj_point, np.ones((obj_point.shape[0],1)))) # make homogeneous
    img_point_est = K.dot(Rt.dot(obj_point.transpose())[0:3])
    img_point_est = img_point_est / img_point_est[2]  # divide by last coordinate
    img_point_est = img_point_est.transpose()
    dist = np.linalg.norm(img_point_gt[:,0:2] - img_point_est[:,0:2], axis=-1)
    indices = [dist < threshold][0].nonzero()[0]
    inliers = matches_for_image[indices, :]
    return inliers, indices

def ransac(matches_for_image, intrinsics):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = MAX_RANSAC_ITERS  # can set this to whatever you want to start with
    k = 0
    max = np.iinfo(np.int32).min
    best_model = {}

    while k < no_iterations:
        k = k + 1

        # pick 4 random matches (assume they are inliers)
        random_matches = np.random.choice(len(matches_for_image), s, replace=False)

        # get 3D and 2D points
        img_points = matches_for_image[(random_matches), 0:2]
        obj_points = matches_for_image[(random_matches), 2:5]

        Rt = model_fit(img_points, obj_points, cv2.SOLVEPNP_P3P, intrinsics)
        matches_without_random_matches = np.delete(matches_for_image, random_matches, axis=0)
        inliers, _ = model_evaluate(matches_without_random_matches, Rt, ERROR_THRESHOLD, intrinsics)

        # TODO: remove this "+s" ?
        inliers_no = len(inliers) + s #total number of inliers
        outliers_no = len(matches_for_image) - inliers_no

        # store best model so far
        if(inliers_no > max):
            best_model['Rt'] = Rt
            best_model['inliers_no'] = inliers_no
            best_model['inliers'] = np.vstack((matches_for_image[(random_matches),:], inliers)) #add the previous random 4 matches too!
            best_model['inliers_for_refit'] = inliers
            best_model['outliers_no'] = outliers_no
            best_model['iterations'] = k
            max = inliers_no
            e = outliers_no / len(matches_for_image)
            N = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
            N = int(np.floor(N))
            no_iterations = N
            if(k > N): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                break #return inliers_no, outliers_no, k, best_model, inliers

        if (k >= MAX_RANSAC_ITERS):
            break #return inliers_no, outliers_no, k, best_model, inliers

    #re-fit here on last inliers set you get..

    # if unable to get pose then just return None
    if(not bool(best_model)):
        return None

    # This will only run if the inlers of the best model are over or equal to 4
    if(best_model['inliers_for_refit'].shape[0] >= 4):
        best_model['Rt'] = model_refit(best_model['inliers_for_refit'][:,0:2], best_model['inliers_for_refit'][:,2:5], intrinsics)

    return best_model

def ransac_dist(matches_for_image, intrinsics):
    s = 4  # or minimal_sample_size
    p = 0.99 # this is a typical value
    # number of iterations (http://www.cse.psu.edu/~rtc12/CSE486/lecture15.pdf and https://youtu.be/5E5n7fhLHEM?list=PLTBdjV_4f-EKeki5ps2WHqJqyQvxls4ha&t=428)
    # also reddit post https://www.reddit.com/r/computervision/comments/gikj1s/can_somebody_help_me_understand_ransac/
    no_iterations = MAX_RANSAC_ITERS  # can set this to whatever you want to start with
    k = 0
    max = np.iinfo(np.int32).min
    best_model = {}
    distribution = matches_for_image[:,-1]

    while k < no_iterations:
        k = k + 1
        # pick 4 random matches (assume they are inliers)
        # p = distribution, From docs: The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a
        random_matches = np.random.choice(len(matches_for_image), s , p = distribution, replace=False)

        # get 3D and 2D points
        img_points = matches_for_image[(random_matches), 0:2]
        obj_points = matches_for_image[(random_matches), 2:5]

        Rt = model_fit(img_points, obj_points, cv2.SOLVEPNP_P3P, intrinsics)
        matches_without_random_matches = np.delete(matches_for_image, random_matches, axis=0)
        inliers, _ = model_evaluate(matches_without_random_matches, Rt, ERROR_THRESHOLD, intrinsics)

        inliers_no = len(inliers) + s #total number of inliers
        outliers_no = len(matches_for_image) - inliers_no

        # store best model so far
        if(inliers_no > max):
            best_model['Rt'] = Rt
            best_model['inliers_no'] = inliers_no
            best_model['inliers'] = np.vstack((matches_for_image[(random_matches),:], inliers)) #add the previous random 4 matches too!
            best_model['inliers_for_refit'] = inliers
            best_model['outliers_no'] = outliers_no
            best_model['iterations'] = k
            max = inliers_no
            e = outliers_no / len(matches_for_image)
            N = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
            N = int(np.floor(N))
            no_iterations = N
            if(k > N ): # this is saying if the max number of iterations you should have run is N, but you already did k > N then no point continuing
                break

        if (k >= MAX_RANSAC_ITERS):
            break

    #re-fit here on last inliers set you get..

    # if unable to get pose then just return None
    if (not bool(best_model)):
        return None

    # This will only run if the inlers of the best model are over or equal to 4
    if(best_model['inliers_for_refit'].shape[0] >= 4):
        best_model['Rt'] = model_refit(best_model['inliers_for_refit'][:,0:2], best_model['inliers_for_refit'][:,2:5], intrinsics)

    return best_model

# for findSupport()
# from emails - with author
# Yes, it is okay.
# A better one is considering also errors of points. For example:
# sum_errors = 0; num_inliers = 0
# for pt in points:
#    error = estimator.getError(model, pt) # e.g., estimator is PnP,
# error function computes reprojection error
#    if error < threshold:
#        sum_errors += error
#        num_inliers += 1
# sum_errors += (len(points) - num_inliers) * threshold
# So, you can compare models either by number of inliers (higher is
# better) or by sum of errors (lower is better).

def prosac(sorted_matches, intrinsics):
    CORRESPONDENCES = sorted_matches.shape[0]
    isInlier = np.zeros([1,CORRESPONDENCES])
    SAMPLE_SIZE = 4
    MAX_OUTLIERS_PROPORTION = 0.9
    P_GOOD_SAMPLE = 0.99
    TEST_NB_OF_DRAWS = MAX_RANSAC_ITERS
    TEST_INLIERS_RATIO = 0.5
    BETA = 0.01
    ETA0 = 0.05
    Chi2value = 2.706

    def niter_RANSAC(p, epsilon, s, Nmax):
        if(Nmax == -1):
            Nmax = np.iinfo(np.int32).max
        if(not (Nmax >= 1)):
            raise RuntimeError("Nmax has to be positive")
        if(epsilon <= 0):
            return 1
        logarg = - np.exp(s * np.log(1 - epsilon))
        logval = np.log(1 + logarg)
        N = np.log(1 - p) / logval
        if(logval < 0 and N < Nmax):
            return np.ceil(N)
        return Nmax

    def Imin(m, n, beta):
        mu = n * beta
        sigma = np.sqrt(n * beta * (1 - beta))
        return np.ceil(m + mu + sigma * np.sqrt(Chi2value))

    # def findSupport(n, isInlier):
    #     #n is N and it is not used here
    #     total_inliers = isInlier.sum() # this can change to a another function (i.e kernel) as it is, it is too simple ?
    #     return total_inliers, isInlier

    N = CORRESPONDENCES
    m = SAMPLE_SIZE
    T_N = niter_RANSAC(P_GOOD_SAMPLE, MAX_OUTLIERS_PROPORTION, SAMPLE_SIZE, -1)
    beta = BETA
    I_N_min = (1 - MAX_OUTLIERS_PROPORTION)*N
    logeta0 = np.log(ETA0)

    n_star = N
    I_n_star = 0
    I_N_best = 0
    t = 0
    n = m
    T_n = T_N

    for i in range(m):
        T_n = T_n * (n - i) / (N - i)

    T_n_prime = 1
    k_n_star = T_N

    best_model = {}

    while (((I_N_best < I_N_min) or t <= k_n_star) and t < T_N and t < TEST_NB_OF_DRAWS):
        # best_model = {} #TODO:(16/07/2020 might not need this line ?)
        t = t + 1

        if ((t > T_n_prime) and (n < n_star)):
            T_nplus1 = (T_n * (n+1)) / (n+1-m)
            n = n + 1
            T_n_prime = T_n_prime + np.ceil(T_nplus1 - T_n)
            T_n = T_nplus1

        if (t > T_n_prime):
            pts_idx = np.random.choice(n, m, replace=False)
        else:
            pts_idx = np.append(np.random.choice(n-1, m-1, replace=False), n-1) # TODO: n-1 will need to change to a function (16/07/2020 what ?)

        sample = sorted_matches[pts_idx]
        img_points = sample[:, 0:2]
        obj_points = sample[:, 2:5]

        Rt = model_fit(img_points, obj_points, cv2.SOLVEPNP_P3P, intrinsics)
        inliers, inliers_indices = model_evaluate(sorted_matches, Rt, ERROR_THRESHOLD, intrinsics)
        isInlier = np.zeros([1, CORRESPONDENCES])
        isInlier[0, inliers_indices] = 1

        # This can change to a another function (i.e kernel, from email) as it is, it is too simple ?
        I_N = isInlier.sum() #support of the model, previously was findSupport().

        if(I_N > I_N_best):
            I_N_best = I_N
            n_best = N
            I_n_best = I_N

            best_model['Rt'] = Rt
            best_model['inliers_no'] = I_N
            best_model['inliers'] = np.vstack((sample, inliers)) #add the previous random 4 matches too!
            best_model['inliers_for_refit'] = inliers
            best_model['outliers_no'] = CORRESPONDENCES - I_N
            best_model['iterations'] = t

            if(1):
                epsilon_n_best = I_n_best / n_best
                I_n_test = I_N
                for n_test in range(N, m, -1):
                    if (not (n_test >= I_n_test)):
                        raise RuntimeError("Loop invariant broken: n_test >= I_n_test")
                    if ( (I_n_test * n_best > I_n_best * n_test) and (I_n_test > epsilon_n_best * n_test + np.sqrt(n_test * epsilon_n_best * (1 - epsilon_n_best) * Chi2value) )):
                        if (I_n_test < Imin(m, n_test, beta)):
                            break
                        n_best = n_test
                        I_n_best = I_n_test
                        epsilon_n_best = I_n_best / n_best
                    I_n_test = I_n_test - isInlier[0, n_test - 1]

            if (I_n_best * n_star > I_n_star * n_best):
                if(not (n_best >= I_n_best)):
                    raise RuntimeError("Assertion not respected: n_best >= I_n_best")
                n_star = n_best
                I_n_star = I_n_best
                k_n_star = niter_RANSAC(1 - ETA0, 1 - I_n_star / n_star, m, T_N)

    # return values for readability (16/07/2020 old ones)
    # inlier_no = I_N
    # outliers_no = len(sorted_matches) - inlier_no
    # iterations = t

    #re-fit here on last inliers set you get..

    # if unable to get pose then just return None
    if (not bool(best_model)):
        return None

    # This will only run if the inlers of the best model are over or equal to 4
    if(best_model['inliers_for_refit'].shape[0] >= 4):
        best_model['Rt'] = model_refit(best_model['inliers_for_refit'][:,0:2], best_model['inliers_for_refit'][:,2:5], intrinsics)

    return best_model
