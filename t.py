# from fast_alpr import ALPR

# # You can also initialize the ALPR with custom plate detection and OCR models.
# alpr = ALPR(
#     detector_model="yolo-v9-t-384-license-plate-end2end",
#     ocr_model="cct-xs-v2-global-model",
# )

# # The "assets/test_image.png" can be found in repo root dir
# alpr_results = alpr.predict("image.png")
# print(alpr_results)

import cv2, numpy as np, sys, time as t, os
from fast_alpr import ALPR
from sort import Sort
from ultralytics import YOLO

path = sys.argv[1]

video_cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG) # cv2.CAP_FFMPEG for rtsp
if not video_cap.isOpened():
    print("couldnt open video stream")
    exit()


model_path = "yolov8n.onnx"
if not os.path.exists(model_path):
    tmp = YOLO("yolov8n.pt")
    tmp.export(format="onnx", imgsz=320, half=False)

model = YOLO("yolov8n.onnx", task="detect")
tracker = Sort()
alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",ocr_model="cct-xs-v2-global-model", ocr_device="cpu")

plate_cache, track_last_seen = {}, {}
frame_idx = 0

def get_dominant_color(crop):
    h, w = crop.shape[:2]
    patch = crop[h//4:3*h//4, w//4:3*w//4]
    if patch.size == 0:
        return "unknown", (128, 128, 128)
    avg = patch.reshape(-1, 3).mean(axis=0)
    b, g, r = map(int, avg)
    return f"rgb({r},{g},{b})", (b, g, r)

# ALPR 
def run_alpr(crop):
    try:
        res = alpr.predict(crop)
        if res and len(res) > 0:
            ocr = res[0].ocr
            if ocr and ocr.text:
                return ocr.text
    except:
        pass
    return ""



while True:
    st = t.time()
    ret, frame = video_cap.read()
    if not ret:
        continue
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    frame = cv2.resize(frame, (640, 640))
    frame_idx += 1
    results = model.predict(frame, imgsz=320, conf=0.4, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            continue
        detections.append([x1, y1, x2, y2, conf])
    detections = np.array(detections) if detections else np.empty((0, 5))

    tracks = tracker.update(detections)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        raw_id = int(track[4])
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        color_name, color_bgr = get_dominant_color(crop)
        track_last_seen[raw_id] = frame_idx
        plate_text = plate_cache.get(raw_id, "")
        # ALPR only if not cached + every few frames
        if not plate_text and frame_idx % 5 == 0:
            plate = run_alpr(crop)
            if plate and len(plate) > 3:
                plate_text = plate
                plate_cache[raw_id] = plate_text

        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(frame, f"ID:{raw_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        cv2.putText(frame, plate_text, (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
        print(f"ID: {raw_id}\t{plate_text}")

    fps = 1.0 / max(t.time() - st, 1e-6)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("traffic", frame)



cv2.destroyAllWindows()
video_cap.release()