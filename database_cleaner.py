from query_image import read_images_binary
from database import COLMAPDatabase

submap_images_path = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/model/0/images.bin"
submap_images = read_images_binary(submap_images_path)
submap_images_names = []

for k, v in submap_images.items():
    submap_images_names.append(v.name)

db = COLMAPDatabase.connect("/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/database.db")
image_names = db.execute("SELECT name FROM images")
image_names = image_names.fetchall()

reference_map_images = []

for name in image_names:
    reference_map_images.append(name[0])

reference_map_images_to_remove = []

for name in reference_map_images:
    if name not in submap_images_names:
        reference_map_images_to_remove.append(name)

cursor = db.cursor()
for i in range(len(reference_map_images_to_remove)):
    print("Removing image " + str(i) + "/" + str(len(reference_map_images_to_remove)), end="\r")

    image_to_remove_name = reference_map_images_to_remove[i]
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + image_to_remove_name + "'")
    image_id = str(image_id.fetchone()[0])

    cursor.execute("DELETE FROM images WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM descriptors WHERE image_id = ?", (image_id,))
    db.commit()

    cursor.execute("DELETE FROM keypoints WHERE image_id = ?", (image_id,))
    db.commit()

print("Done")
