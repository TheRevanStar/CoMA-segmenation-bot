import argparse
from data_prepare import extract_classes_from_annotation,create_hidden_folder,FILE_ATTRIBUTE_HIDDEN
from ultralytics import YOLO
import os
import yaml
import ctypes
import datetime
import torch
import torchvision
from clearml import Task
import glob


model_variant = "yolo11s"


def get_latest_run():
    runs = glob.glob('runs/train/*')
    runs = [d for d in runs if os.path.isdir(d)]
    if not runs:
        return None
    return max(runs, key=os.path.getmtime)




model = YOLO(f"{model_variant}.pt")

def data_to_yaml( train_path, val_path, classes):
    data_dict={
         'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': classes
    }
    
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_filename=f"dataset_{time_str}.yaml"
    folder_path =os.path.join(os.getcwd(), 'data_yaml')
    create_hidden_folder(folder_path)
    yaml_path = os.path.join(folder_path, yaml_filename)
    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f)
    return yaml_path

def model_train(path_to_train='',path_to_val='',EPOCHS=50,BATCH_SIZE=32,IMGSZ=640,device='cuda',classes=None,resume=False):
    if not classes:
        print('Error class extraction failure')
        classes=[]
    data_yaml_path=data_to_yaml( path_to_train, path_to_val, classes)
    args = dict(data= data_yaml_path, 
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        patience=5,
        resume=resume,
        workers=8)
    task.connect(args)
   
    result=model.train(**args)
    return result


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser=argparse.ArgumentParser('parse args for training the model')
    parser.add_argument('--path_to_csv',type=str,required=True, help='Path to annotation CSV file')
    parser.add_argument('--device',type=str,default='cuda' , help='Path to annotation CSV file')
    parser.add_argument('--batch_size',type=int,default=32,help='size of batch to work')
    parser.add_argument('--imgsz_size',type=int,default=480,help='size of image')
    parser.add_argument('--epochs',type=int,default=50,help='epoch count')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    args=parser.parse_args()

    task = Task.init(project_name="MoCA_segmentation", task_name="YOLO11_train"+time_str)
    task.set_parameter("model_variant", model_variant)

    classes_list = list(extract_classes_from_annotation(args.path_to_csv))
    latest_run = get_latest_run()
    if latest_run:
        checkpoint_path = os.path.join(latest_run, 'weights', 'last.pt')

    if args.resume and os.path.exists(checkpoint_path):
        model = YOLO(checkpoint_path)
        latest_run = get_latest_run()
        if latest_run:
             checkpoint_path = os.path.join(latest_run, 'weights', 'last.pt')
    else:
        model = YOLO(f"{model_variant}.pt")

    model_train(path_to_train=os.path.join(os.getcwd(),r'dataset\train\images'),
                path_to_val=  os.path.join(os.getcwd(),r'dataset\val\images'),
               EPOCHS=args.epochs,
               BATCH_SIZE=args.batch_size ,
                device=args.device,
                IMGSZ=args.imgsz_size,
                classes=classes_list,
                resume=args.resume
                
                )
    