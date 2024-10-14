import gc
import multiprocessing
from ultralytics import YOLO
import optuna
import torch
import os
import numpy as np

def objective(trial):
    # Define the hyperparameters to optimize
    epochs = trial.suggest_int('epochs', 10, 100)
    batch_size = trial.suggest_categorical('batch_size', [-1, 10, 30])
    img_size = trial.suggest_int('img_size', 640, 1280)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)

    # Load a pretrained YOLO model
    model = YOLO(model=MODEL_NAME, task='detect')

    # Train the model
    model.train(
        data=YAML_PATH,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate
    )

    # Evaluate the model performance on the validation set and return a metric to optimize
    results = model.val()
    # Use mAP(0.5) as the metric to optimize
    return results.box.map50  # Adjust this based on the actual results structure


YAML_PATH = r"D:/Image_Data/image_data/data.yaml"
YAML_ROBO_PATH = r"D:/Image_Data/image_data/argumentation_Robo/defect_yolov8/data.yaml"
TEST_PATH = r"D:/Image_Data/image_data/test/data.yaml"
AGUMENTAION_PATH=r"D:/Image_Data/image_data/hyp.scratch.yaml"

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["TORCH_USE_CUDA_DSA"] = '1'

    torch.__version__

    print("np version : " + np.__version__)

    gc.collect()
    torch.cuda.init()
    torch.cuda.empty_cache()
    multiprocessing.freeze_support()  # windows에서는 필수
    #C:/Users/ourdr/AppData/Roaming/Ultralytics/settings.yaml
    '''
        # Create an Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)

        # Print the best trial
        print('Best trial:')
        trial = study.best_trial
        print(f'  Value: {trial.value}')
        print('  Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')

        # Load the best hyperparameters and train the final model
        best_params = trial.params

        # Load a model
        # model = YOLO("yolov8n.yaml") # build a new model from scratch
        model = YOLO(model=MODEL_NAME, task='detect')  # load a pretrained model (recommended for training)
        model.train(
            data=YAML_PATH,
            epochs=best_params['epochs'],
            batch=best_params['batch_size'],
            imgsz=best_params['img_size'],
            lr0=best_params['learning_rate']
        )
        '''
    # Load a model
    MODEL_NAME = r'yolov8s.pt'
    #MODEL_NAME = r'yolo11n.pt'
    model = YOLO(model=MODEL_NAME, task='detect')  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=YAML_PATH,
                epochs=100,
                batch=10,
                amp=True,
                retina_masks=True,
                auto_augment=False,
                augment=False,
                mixup=0.0,
                mosaic=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                flipud=0.0,
                fliplr=0.0,
                degrees=0.0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                crop_fraction=1.0,
                erasing=0.0,
                copy_paste=0.0,
                nms=True,
                cos_lr=True,
                dropout=0.3,
                iou=0.5,
                imgsz=1024)  # train the model

    model.val()  # evaluate model performance on the validation set

    model(r"D:/Image_Data/image_data/raw_Data/Final_Data/split_images_Final_241012/0.Stain/20603F00818_STAIN(3179 X 1093)_C15GX_1_3.jpg")  # predict on an image
    #success = model.export(format='onnx')  # export the model to ONNX format
