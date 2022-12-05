import os
import shutil
import random

random.seed(7)


def generate(imgs_path, labels_path, gen_target='./', test_rate=0.00, val_rate=0.08):
    train_folder = os.path.join(gen_target, 'train')
    test_folder = os.path.join(gen_target, 'test')
    val_folder = os.path.join(gen_target, 'val')

    train_folder_images = os.path.join(train_folder, 'images')
    train_folder_labels = os.path.join(train_folder, 'labels')

    test_folder_images = os.path.join(test_folder, 'images')
    test_folder_labels = os.path.join(test_folder, 'labels')

    val_folder_images = os.path.join(val_folder, 'images')
    val_folder_labels = os.path.join(val_folder, 'labels')

    if not os.path.exists(gen_target):
        os.mkdir(gen_target)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)

    if not os.path.exists(train_folder_images):
        os.mkdir(train_folder_images)
    if not os.path.exists(train_folder_labels):
        os.mkdir(train_folder_labels)
    if not os.path.exists(test_folder_images):
        os.mkdir(test_folder_images)
    if not os.path.exists(test_folder_labels):
        os.mkdir(test_folder_labels)
    if not os.path.exists(val_folder_images):
        os.mkdir(val_folder_images)
    if not os.path.exists(val_folder_labels):
        os.mkdir(val_folder_labels)

    labels = []
    for file in os.listdir(labels_path):
        try:
            if os.path.basename(file).split('.')[1] == 'txt':
                labels.append(file)
        except Exception as r:
            print(r)

    count = len(labels)
    test_count = int(test_rate * count)
    val_count = int(val_rate * count)

    random.shuffle(labels)
    test_files = labels[:test_count]
    val_files = labels[test_count: test_count + val_count]
    train_files = labels[test_count + val_count:]

    for item in test_files:
        basename = os.path.basename(item).split('.')[0]
        src_label_file = os.path.join(labels_path, basename + '.txt')
        dst_label_file = os.path.join(test_folder_labels, basename + '.txt')
        shutil.copy(src_label_file, dst_label_file)

        src_image_file = os.path.join(imgs_path, basename + '.jpg')
        if not os.path.exists(src_image_file):
            continue
        dst_label_file = os.path.join(test_folder_images, basename + '.jpg')
        shutil.copy(src_image_file, dst_label_file)

    for item in val_files:
        basename = os.path.basename(item).split('.')[0]
        src_label_file = os.path.join(labels_path, basename + '.txt')
        dst_label_file = os.path.join(val_folder_labels, basename + '.txt')
        shutil.copy(src_label_file, dst_label_file)

        src_image_file = os.path.join(imgs_path, basename + '.jpg')
        dst_label_file = os.path.join(val_folder_images, basename + '.jpg')
        shutil.copy(src_image_file, dst_label_file)

    for item in train_files:
        basename = os.path.basename(item).split('.')[0]
        src_label_file = os.path.join(labels_path, basename + '.txt')
        dst_label_file = os.path.join(train_folder_labels, basename + '.txt')
        shutil.copy(src_label_file, dst_label_file)

        src_image_file = os.path.join(imgs_path, basename + '.jpg')
        dst_label_file = os.path.join(train_folder_images, basename + '.jpg')
        shutil.copy(src_image_file, dst_label_file)


generate('/data/jack_ssd/home/jack/tunm/work/training_lp_det/train_all/images',
         '/data/jack_ssd/home/jack/tunm/work/training_lp_det/train_all/labels',
         '/data/jack_ssd/home/jack/tunm/work/training_lp_det/training_yolov5')
