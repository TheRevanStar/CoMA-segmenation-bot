import os
import csv
import ast
import cv2
import argparse
import ctypes
import shutil
import datetime
from sklearn.model_selection import train_test_split
import random 

TRAIN_RATIO = 0.75
TEST_RATIO = 0.05
VAL_RATIO = 0.2
FILE_ATTRIBUTE_HIDDEN = 0x02
CSV_FIELDNAMES = [
    'id', 'file_list', 'unknown1', 'unknown2', 'spatial_coordinates', 'metadata',
    'col7', 'col8', 'col9', 'col10'
]

def create_hidden_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        ret = ctypes.windll.kernel32.SetFileAttributesW(path, FILE_ATTRIBUTE_HIDDEN)
        if not ret:
            raise ctypes.WinError()
        
        
def split_data(source_dir, train_dir, val_dir, test_dir, seed=42):
    files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    train_val_files, test_files = train_test_split(
        files, test_size=TEST_RATIO, random_state=seed, shuffle=True)

    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size_adjusted, random_state=seed, shuffle=True)

    def copy(file_list, dist_dir):
        os.makedirs(dist_dir, exist_ok=True)
        for src_path in file_list:
            dst_path = os.path.join(dist_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)

    copy(train_files, train_dir)
    copy(val_files, val_dir)
    copy(test_files, test_dir)


def yolo_box_converter(img_width, img_height, x, y, w, h):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def extract_classes_from_annotation(annotation_path, skip_lines=0):
    classes_set = set()
    with open(annotation_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                file_path = row[1].strip()
                if '/' in file_path:
                    class_name = file_path.split('/')[1]
                    classes_set.add(class_name)
    return sorted(classes_set)





def process_single_folder(img_path, annotation, classes_list, buffer_label_dir, buffer_img_dir, skip_lines=0):
    os.makedirs(buffer_img_dir, exist_ok=True)
    os.makedirs(buffer_label_dir, exist_ok=True)

    classes = sorted(list(set(classes_list)))
    id_dict = {cls: i for i, cls in enumerate(classes)}

    saved_images = 0
    saved_labels = 0

    counters = {} 

    with open(annotation, newline='', encoding='utf-8') as csvfile:
        for _ in range(skip_lines):
            next(csvfile)
        reader = csv.DictReader(csvfile, fieldnames=CSV_FIELDNAMES)
        for idx, row in enumerate(reader, 1):
            file_list = row.get('file_list', '').strip()
            if not file_list:
                continue

            relative_path = file_list.lstrip('/').replace('/', os.sep).replace('\\', os.sep)
            image_path = os.path.join(img_path, relative_path)

            if not os.path.exists(image_path):
                print(f"[{idx}] Image not found: {image_path}")
                continue

            dir_name = relative_path.split(os.sep)[0]

            if dir_name not in id_dict:
                print(f"[{idx}] Class '{dir_name}' not in classes list, skipping")
                continue
            class_id = id_dict[dir_name]

          
            if dir_name not in counters:
                counters[dir_name] = 1
            else:
                counters[dir_name] += 1

            img = cv2.imread(image_path)
            if img is None:
                print(f"[{idx}] Could not read image: {image_path}")
                continue

            try:
                coords = ast.literal_eval(row.get('spatial_coordinates', ''))
            except Exception:
                continue
            if not coords or coords[0] != 2:
                continue
            _, x, y, w, h = coords

            img_h, img_w = img.shape[:2]
            x_box, y_box, w_box, h_box = yolo_box_converter(img_w, img_h, x, y, w, h)

            if not (0 <= x_box <= 1 and 0 <= y_box <= 1 and 0 <= w_box <= 1 and 0 <= h_box <= 1):
                continue

            img_resized = cv2.resize(img, (640, 640))
            if img_resized is None or img_resized.size == 0:
                print(f"[{idx}] Empty resized image for {file_list}")
                continue

          
            image_basename = f"{dir_name}_{counters[dir_name]:06d}"

            save_img_path = os.path.join(buffer_img_dir, image_basename + '.jpg')
            print(f"[{idx}] Saving image to: {os.path.abspath(save_img_path)}")
            success = cv2.imwrite(save_img_path, img_resized)
            if not success:
                print(f"[{idx}] Failed to save image: {save_img_path}")
                continue
            saved_images += 1

            label_filename = image_basename + '.txt'
            label_path = os.path.join(buffer_label_dir, label_filename)
            print(f"[{idx}] Saving label to: {os.path.abspath(label_path)}")
            with open(label_path, 'a') as f_label:
                f_label.write(f"{class_id} {x_box:.6f} {y_box:.6f} {w_box:.6f} {h_box:.6f}\n")
            saved_labels += 1

    print(f"Total images saved: {saved_images}")
    print(f"Total labels saved: {saved_labels}")


          




def data_prepare(img_path='', annotation='', skip_lines=0):
    classes_list = extract_classes_from_annotation(annotation, skip_lines=skip_lines)
    print(f"Detected classes: {classes_list}")

    project_dir = os.getcwd()
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    buffer_label_dir = os.path.join(project_dir, 'dataset', 'buffer_label_' + time_str)
    buffer_img_dir = os.path.join(project_dir, 'dataset', 'buffer_img_' + time_str)

    create_hidden_folder(buffer_label_dir)
    create_hidden_folder(buffer_img_dir)

    print(f"Processing folder: {img_path}")
    process_single_folder(img_path, annotation, classes_list, buffer_label_dir, buffer_img_dir, skip_lines=skip_lines)

    train_dir = os.path.join(project_dir, 'dataset', 'train')
    val_dir = os.path.join(project_dir, 'dataset', 'val')
    test_dir = os.path.join(project_dir, 'dataset', 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    split_data(buffer_img_dir, os.path.join(train_dir, 'images'), os.path.join(val_dir, 'images'), os.path.join(test_dir, 'images'))
    split_data(buffer_label_dir, os.path.join(train_dir, 'labels'), os.path.join(val_dir, 'labels'), os.path.join(test_dir, 'labels'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    parser.add_argument('--img_path', type=str, required=True, help='Path to images root directory')
    parser.add_argument('--annotation', type=str, required=True, help='Path to annotation CSV file')
    parser.add_argument('--skip_lines', type=int, default=0, help='Number of lines to skip before CSV header')

    args = parser.parse_args()

    data_prepare(
        img_path=args.img_path,
        annotation=args.annotation,
        skip_lines=args.skip_lines
    )
