import os
import random
import shutil
data_path = './plant-seedlings-classification/train'
classes = os.listdir(data_path)

test_path = './plant-seedlings-classification/mytest'
# create class folders
for c in classes:
    os.mkdir(os.path.join(test_path, c))

for c in classes:
    src_path = os.path.join(data_path, c)
    dest_path = os.path.join(test_path, c)
    files = os.listdir(src_path)
    no_of_files = len(files) // 5

    for file_name in random.sample(files, no_of_files):
        shutil.move(os.path.join(src_path, file_name), dest_path)