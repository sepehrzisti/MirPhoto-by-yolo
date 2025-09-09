if __name__ == "__main__":
    from ultralytics import YOLO
    import torch
    import os

    # بررسی GPU
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # مسیر فایل data.yaml که شامل train/val/test است
    data_yaml = "C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/data.yaml"

    # مدل پایه pretrained (COCO)
    model = YOLO("yolov8n.pt")

    # --- Train ---
    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=2,
        name="bone_fractures",
        device="0",
        augment=True,
        workers=0
    )

    # مسیر بهترین مدل بعد از train
    best_model_path = "C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/runs/detect/bone_fractures/weights/best.pt"

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    # --- Load بهترین مدل ---
    trained_model = YOLO(best_model_path)

    # --- Inference روی تصاویر test ---
    test_images_path = "C:/Users/sepehr/Desktop/projects/MRI/Bone Fractures Detection/test/images"
    results = trained_model.predict(
        source=test_images_path,
        conf=0.25,
        save=True
    )

    # نمایش اولین نتیجه به عنوان نمونه
    results[0].show()

    print(f"نتایج روی test در پوشه: runs/detect/predict ذخیره شد")