import multiprocessing
import optuna
from ultralytics import YOLO

def objective(trial):
    # Define the hyperparameters to optimize
    epochs = trial.suggest_int('epochs', 10, 20)
    batch_size = trial.suggest_categorical('batch_size', [-1, 10, 16])
    img_size = trial.suggest_int('img_size', 600, 800)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Load a pretrained YOLO model
    model = YOLO(model='yolov8s.pt', task='detect')

    # Train the model
    model.train(
        data=r"C:\\Users\\Chan's Victus\\Desktop\\pythonProject\\capstonD\\data\\dataset\\data.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate
    )

    # Evaluate the model performance on the validation set and return a metric to optimize
    results = model.val()
    # Use mAP(0.5) as the metric to optimize
    return results.box.map50  # Adjust this based on the actual results structure

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Windows에서는 필수

    # Create an Optuna study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    # Print the best trial
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Load the best hyperparameters and train the final model
    best_params = trial.params
    model = YOLO(model='yolov8n.pt', task='detect')
    model.train(
        data=r"C:\\Users\\Chan's Victus\\Desktop\\pythonProject\\capstonD\\data\\dataset\\data.yaml",
        epochs=best_params['epochs'],
        batch=best_params['batch_size'],
        imgsz=best_params['img_size'],
        lr0=best_params['learning_rate']
    )

    # Evaluate the final model performance on the validation set
    model.val()

    # Predict on an image
    model(r"C:\Users\Chan's Victus\Desktop\pythonProject\capstonD\data\dataset\images\train\MVI_0788_VIS_frame0.jpg")

    # Export the model to ONNX format
    success = model.export(format='onnx')
