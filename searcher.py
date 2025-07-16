
from ultralytics import YOLO
import os
from clearml import Task
import cv2
import numpy as np
from test import model_predict
import ctypes
from data_prepare import FILE_ATTRIBUTE_HIDDEN,create_hidden_folder
import wikipedia
import re

filter_list = [
    'animal',
    'bird',
    'fish',
    'mammal',
    'reptile',
    'amphibian',
    'insect',
    'arthropod',
    'crustacean',
    'mollusk',
    'marine',
    'fauna',
    'herpetology',  
    'avian',        
    'vertebrate',
    'invertebrate',
]



def filter_pages(search_results):
    filtered_result = []
    other_result = []

    for title in search_results:
        try:
            page = wikipedia.page(title, auto_suggest=True)
            categories = page.categories 

           
            if any(keyword in cat.lower() for cat in categories for keyword in filter_list):
                filtered_result.append(page)
            else:
                other_result.append(title)

        except Exception as e:
            other_result.append((title, str(e)))

    return filtered_result, other_result
    
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
    main_results_obj = results[0][0]
    confidences = main_results_obj.boxes.conf.cpu().numpy()
    best_confidence_idx = np.argmax(confidences)
    class_id = int(main_results_obj.boxes.cls[best_confidence_idx].cpu().numpy())
    
    raw_class_name = model.names[class_id]

    wikipedia.set_lang("en")

    clean_name = re.sub(r'[\d_]+', ' ', raw_class_name)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()

    result_search = ""
    other_result_search = ""

    try:
        search_results = wikipedia.search(clean_name)
        if search_results:
            result_filtered_search, other_result = filter_pages(search_results)
           
            for page in result_filtered_search:
                result_search += f"Title: {page.title}\nURL: {page.url}\n\n"

            other_result_search = ''
            for item in other_result:
                if not isinstance(item, tuple):

                    other_result_search += f"Not matched: {item}\n"

        else:
            result_search = "Page not found."

    except wikipedia.PageError:
        result_search = "Page not found."
    except Exception as e:
        result_search = f"Error occurred: {e}"

    return result_search, other_result_search
