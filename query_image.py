import time

import cv2
import numpy as np
import struct
import collections

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# TODO: replace this method with the text version and get the last one ?
def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def get_query_image_global_pose(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/0/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec2rotmat()
    pose_t = image.tvec
    pose = np.c_[pose_r, pose_t]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def get_image_camera_center_by_name(name, images):
    cam_center = np.array([])
    for k,v in images.items():
        if(v.name == name):
            pose_r = v.qvec2rotmat()
            pose_t = v.tvec
            pose = np.c_[pose_r, pose_t]
            pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
            rot = np.array(pose[0:3, 0:3])
            cam_center = -rot.transpose().dot(pose_t)
    return cam_center

def get_images_camera_quats(images):
    quats = {}
    for k,v in images.items():
        quat = v.qvec
        quats[v.name] = quat
    return quats

def get_images_camera_centers(images):
    cam_centers = {}
    for k,v in images.items():
        pose_r = v.qvec2rotmat()
        pose_t = v.tvec
        pose = np.c_[pose_r, pose_t]
        pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
        rot = np.array(pose[0:3, 0:3])
        cam_center = -rot.transpose().dot(pose_t)
        cam_centers[v.name] = cam_center
    return cam_centers

def assign_K_to_frame(images, cameras_path):
    Ks = {}
    for k,v in images.items():
        K = get_intrinsics_from_camera_bin(cameras_path, v.camera_id)
        Ks[v.name] = K
    return Ks

def get_images_camera_principal_axis_vectors(images, Ks):
    principal_axis_vectors = {}
    for k,v in images.items():
        pose_r = v.qvec2rotmat()
        K = Ks[v.name]
        M = K @ pose_r
        m3 = M[2,:]
        principal_axis_vector = np.linalg.det(M) * m3
        principal_axis_vectors[v.name] = principal_axis_vector
    return principal_axis_vectors

def image_localised(name, images):
    image_id = None
    for k, v in images.items():
        if (v.name == name):
            image_id = v.id
            return image_id
    return image_id

# This was copied from feature_matcher_single_image.py
def get_queryDescriptors(db, image_id):
    query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
    query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
    query_image_descriptors_data = db.blob_to_array(query_image_descriptors_data, np.uint8)
    descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
    query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

    row_sums = query_image_descriptors_data.sum(axis=1)
    query_image_descriptors_data = query_image_descriptors_data / row_sums[:, np.newaxis]
    queryDescriptors = query_image_descriptors_data.astype(np.float32)
    return queryDescriptors

# This was copied from feature_matcher_single_image.py
def get_image_id(db, query_image):
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + query_image + "'")
    image_id = str(image_id.fetchone()[0])
    return image_id

# This was copied from feature_matcher_single_image.py
def get_keypoints_xy(db, image_id):
    query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
    query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
    query_image_keypoints_data = db.blob_to_array(query_image_keypoints_data, np.float32)
    query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
    query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows, query_image_keypoints_data_cols)
    query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
    return query_image_keypoints_data_xy

"""
The reconstructed pose of an image is specified as the projection 
from world to the camera coordinate system of an image using a quaternion (QW, QX, QY, QZ) 
and a translation vector (TX, TY, TZ). 
"""
def get_query_image_global_pose_new_model(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec2rotmat()
    pose_t = image.tvec
    pose = np.c_[pose_r, pose_t]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def get_query_images_pose_from_images(names, images):
    images_name_pose = {}
    for name in names:
        pose = get_query_image_pose_from_images(name, images)
        images_name_pose[name] = pose
    return images_name_pose

def get_query_image_pose_from_images(name, images):
    image = None
    for k,v in images.items():
        if v.name == name:
            image = v
    if(image == None):
        return np.array([])
    pose_r = image.qvec2rotmat()
    pose_t = image.tvec
    pose = np.c_[pose_r, pose_t]
    return pose

def get_query_image_global_pose_new_model_quaternion(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    pose_r = image.qvec
    pose_t = image.tvec
    pose = np.r_[pose_t, [pose_r[1],pose_r[2],pose_r[3]],pose_r[0]]
    return pose

def get_query_image_id(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/model/0/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    id = image.id
    return id

def get_query_image_id_new_model(name):
    images = read_images_binary("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/new_model/images.bin")
    for k,v in images.items():
        if v.name == name:
            image = v
    id = image.id
    return id

def get_images_names_from_sessions_numbers(sessions_numbers, db, model_all_images):
    images_names = []
    if(len(sessions_numbers) == 1):
        for k in range(sessions_numbers[0]):
            id = k + 1  # to match db ids
            image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'")
            image_name = str(image_name.fetchone()[0])
            if (image_localised(image_name, model_all_images) != None):
                images_names.append(image_name)
        return images_names
    else:
        images_traversed = 0
        for i in range(len(sessions_numbers)):
            no_images = sessions_numbers[i]
            start = images_traversed
            end = start + no_images
            for k in range(start, end):
                id = k + 1  # to match db ids
                image_name = db.execute("SELECT name FROM images WHERE image_id = " + "'" + str(id) + "'")
                image_name = str(image_name.fetchone()[0])
                if (image_localised(image_name, model_all_images) != None):
                    images_names.append(image_name)
                images_traversed += 1
        return images_names

def get_all_images_names_from_db(db):
    image_names = db.execute("SELECT name FROM images")
    image_names_tuples = image_names.fetchall()
    image_names = [image_names_tuple[0] for image_names_tuple in image_names_tuples]
    return image_names

def get_images_names_bin(images_bin_path):
    images_names = []
    images = read_images_binary(images_bin_path)
    for k,v in images.items():
        images_names.append(v.name)
    return images_names

def get_images_ids(image_names, all_images):
    image_ids = []
    for name in image_names:
        id = image_localised(name, all_images)
        image_ids.append(id)
    return image_ids

def get_images_names(all_images):
    image_names = []
    for k,v in all_images.items():
        image_names.append(v.name)
    return image_names

def load_images_from_text_file(path):
    images = []
    with open(path) as f:
        images = f.readlines()
    images = [x.strip() for x in images]
    return images

# This will take a list of images, check which are localised and
# returns only those. It could be a subset of images_bin or all
def get_localised_image_by_names(names, images_bin_path):
    images = read_images_binary(images_bin_path)
    localised_images = []
    for name in names:
        if(image_localised(name, images) != None):
            localised_images.append(name)
    return localised_images

def get_image_by_name(name, images):
    image = None
    for k, v in images.items():
        if (v.name == name):
            image = v
            return image
    return image

# maybe these shoudln't be here... anyway..
def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def get_intrinsics_from_camera_bin(cameras_path, id):
    cameras = read_cameras_binary(cameras_path)
    camera_params = cameras[id].params
    K = None
    if(camera_params.size == 3):
        fx = camera_params[0]
        fy = camera_params[0]
        cx = camera_params[1]
        cy = camera_params[2]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    if(camera_params.size == 4):
        fx = camera_params[0]
        fy = camera_params[1]
        cx = camera_params[2]
        cy = camera_params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    assert(K is not None)
    return K

def save_image_projected_points(image_path, K, P, points3D, outpath):
    image = cv2.imread(image_path)
    points = K.dot(P.dot(points3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.imwrite(outpath, image)

def save_heatmap_of_image(image_path, K, P, points3D, outpath, values):
    values *= (255.0/values.max())
    points3D[:,3] = 1
    image = cv2.imread(image_path)
    width =  image.shape[1]
    height =  image.shape[0]
    heatmap_image = np.ones((height, width, 1), np.uint8)
    points = K.dot(P.dot(points3D.transpose())[0:3,:])
    points = points / points[2]
    points = points.transpose()
    for i in range(len(points)): #points will have the same order as values_norm, and as points3D
        x = int(points[i][0])
        y = int(points[i][1])
        rgb_val = values[i]
        center = (x, y)
        cv2.circle(image, center, 7, (0, 0, 255), 2)
        cv2.circle(image, center, 5, (rgb_val, rgb_val, rgb_val), -1)
    cv2.imwrite(outpath, image)
