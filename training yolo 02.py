from ultralytics import YOLO

if __name__ == "__main__":

    # -------------------------------
    # Load a STRONG base model
    # -------------------------------
    model = YOLO("yolov8m.pt")   # use yolov8l.pt if GPU allows

    model.train(

        # -------------------------------
        # Dataset
        # -------------------------------
        data="Datasets/Potato_Disease/data.yaml",

        # -------------------------------
        # Training strategy
        # -------------------------------
        epochs=150,              # long convergence
        imgsz=832,               # HIGH resolution for small lesions
        batch=-1,                # auto batch (max GPU usage)
        device=0,                # GPU id
        workers=8,

        # -------------------------------
        # Project settings
        # -------------------------------
        project="Model/PotatoDisease",
        name="YOLOv8_HighAccuracy",
        exist_ok=True,

        # -------------------------------
        # Optimizer (accuracy-oriented)
        # -------------------------------
        optimizer="AdamW",
        lr0=0.0007,              # lower LR for texture learning
        lrf=0.01,
        weight_decay=0.0003,
        warmup_epochs=5,
        cos_lr=True,
        patience=30,

        # -------------------------------
        # Disease-aware augmentations
        # -------------------------------
        mosaic=0.1,              # VERY low (avoid fake leaves)
        mixup=0.0,               # ❌ harmful for disease
        copy_paste=0.0,          # ❌ biologically incorrect

        translate=0.05,
        scale=0.2,
        shear=0.0,

        hsv_h=0.015,
        hsv_s=0.25,
        hsv_v=0.25,

        fliplr=0.5,
        flipud=0.0,

        erasing=0.05,            # light regularization
        close_mosaic=40,

        # -------------------------------
        # Performance & stability
        # -------------------------------
        cache="ram",
        verbose=True,
        save_period=-1,
    )

    # -------------------------------
    # Validation
    # -------------------------------
    metrics = model.val()
    print(metrics)
