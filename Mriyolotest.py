if __name__ == "__main__":
    from ultralytics import YOLO
    import torch


    data_yaml = "C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/data.yaml"
    model = YOLO("C:/Users/sepehr/Desktop/New folder (2)/runs/detect/bone_fractures2/weights/best.pt")


    results = model.predict(source="C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/train/images", conf=0.25, save=True)
    results[5].show()
