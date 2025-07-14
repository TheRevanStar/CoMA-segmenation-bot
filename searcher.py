
from ultralytics import YOLO
import os
from clearml import Task
import cv2
from test import model_predict
import ctypes
from data_prepare import FILE_ATTRIBUTE_HIDDEN,create_hidden_folder
import wikipedia
import re


def search(file_path, model):
    tasks = Task.get_tasks(project_name='MoCA_segmentation')
    if not tasks:
        return "No tasks found in project."

    latest_task = tasks[0]

    path = os.path.join(os.getcwd(), 'tg_img_cache', latest_task.name)
    os.makedirs(path, exist_ok=True)

    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    latest_file = file_path

    results, path_to_img, boxes = model_predict(model, latest_file)

    if boxes is None or len(boxes) == 0:
        return None  

    if isinstance(path_to_img, list):
        path_to_img = path_to_img[0]

    img = cv2.imread(path_to_img)
    x1, y1, x2, y2 = boxes[0].astype(int)
    cropped_img = img[y1:y2, x1:x2]

    path_to_crop = os.path.join(os.getcwd(), 'tg_crop_cache', latest_task.name)
    create_hidden_folder(path_to_crop)

    crop_filename = f"cropped_{len(files)}.jpg"
    cv2.imwrite(os.path.join(path_to_crop, crop_filename), cropped_img)

    class_id = int(results[0][0].boxes.cls[0].cpu().numpy())
    raw_class_name = model.names[class_id]

    wikipedia.set_lang("ru")

    clean_name = re.sub(r'[\d_]+', ' ', raw_class_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    result_search = ""

    try:
        search_results = wikipedia.search(clean_name)
        if search_results:
            pages = wikipedia.page(search_results)
            for page in pages:
                result_search += f"page title: {page.title}\n"
                result_search += f"page url: {page.url}\n"
                result_search += f"short summary:\n{wikipedia.summary(page.title, sentences=3)}"
                result_search += '\n'
                result_search += '\n'
                result_search += '\n'
        else:
            result_search = "Page not found."
    except wikipedia.DisambiguationError as e:
        result_search = f"Many variants found: {e.options}"
    except wikipedia.PageError:
        result_search = "Page not found."

    return result_search
