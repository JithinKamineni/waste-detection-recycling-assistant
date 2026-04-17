from ultralytics import YOLO
from PIL import Image


class GarbageDetector:
    def __init__(self, model_path: str = "Garbage-detection.pt") -> None:
        self.model = YOLO(model_path)

    def detect(self, image_path: str, conf: float = 0.25):
        results = self.model.predict(
            source=image_path,
            conf=conf,
            save=False,
            verbose=False
        )

        result = results[0]
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_rgb)

        detected = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = result.names[class_id].upper()
                detected.append((class_name, confidence))

        return detected, annotated_image