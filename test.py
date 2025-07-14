import argparse
from ultralytics import YOLO
import os
from clearml import Task
import cv2

def model_predict(model,input_path:str) -> str:

    tasks = Task.get_tasks(project_name='MoCA_segmentation')
    output_folder = os.path.join(os.getcwd(), 'result', tasks[0].name)
    os.makedirs(output_folder, exist_ok=True)  

    saved_files = []
    result=[]
    boxes=None
    if os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            img_path = os.path.join(input_path, filename)
            results = model.predict(img_path)
            img_with_boxes = results[0].plot()
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, cv2.resize(img_with_boxes,(640,640)))
            saved_files.append(save_path)
            result.append(results)
    else:
        results = model.predict(input_path)
        img_with_boxes = results[0].plot()
        filename = os.path.basename(input_path)
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, img_with_boxes)
        saved_files.append(save_path)
        boxes=results[0].boxes.xyxy.cpu().numpy()
        result.append(results)

    return result,saved_files ,boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser('parse args for testing the model')
    parser.add_argument('--path_to_img', type=str, required=True, help='path to your test image or directory of images')
    args = parser.parse_args()

    if not os.path.exists(args.path_to_img):
        raise FileNotFoundError(f"path: {args.path_to_img} is incorrect")

    model = YOLO(r'E:\projects\object_deleter\runs\detect\train3\weights\best.pt')

    saved_files = model_predict( model,args.path_to_img)
  