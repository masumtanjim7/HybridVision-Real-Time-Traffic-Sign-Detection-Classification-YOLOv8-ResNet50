import os
import time
import yaml
import cv2
import numpy as np
import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50

# --- YOLOv7 imports (from cloned repo) ---

import sys
sys.path.insert(0, "yolov7")  # make sure yolov7 is first

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox


# Optional webcam support
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.set_page_config(page_title="Traffic Sign Detector (YOLOv7 + ResNet50)", layout="wide")

# ----------------------------
# Config
# ----------------------------
YOLO_WEIGHTS = "weights/best.pt"
RESNET_WEIGHTS = "weights/resnet50_best.pth"
CLASS_NAMES_YAML = "class_names.yaml"

IMG_SIZE = 640
YOLO_CONF_THRES = 0.25
YOLO_IOU_THRES = 0.45
MAX_DET = 100

# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource
def load_class_names(path: str):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    names = y["names"]
    return names

@st.cache_resource
def load_yolo(weights_path: str, device_str: str):
    device = torch.device(device_str)
    model = attempt_load(weights_path, map_location=device)
    model.eval()
    return model, device

class ResNetHead(nn.Module):
    """ResNet50 adapted for your N classes."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = resnet50(weights=None)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
@st.cache_resource
def load_resnet(weights_path: str, num_classes: int, device_str: str):
    device = torch.device(device_str)

    model = ResNetHead(num_classes=num_classes).to(device).eval()
    ckpt = torch.load(weights_path, map_location=device)

    # ✅ your .pth contains {"model_state": ..., "classes": ...}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt  # already a state_dict

    # ✅ remove "module." prefix (if trained with DataParallel)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)

    # (optional) class names saved in checkpoint
    classes = ckpt.get("classes") if isinstance(ckpt, dict) else None
    return model, device, classes


resnet_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def yolo_infer_bgr(img_bgr: np.ndarray, yolo_model, device, conf_thres, iou_thres):
    """
    Returns detections in original image coords:
    list of [x1,y1,x2,y2, conf, cls]
    """
    img0 = img_bgr.copy()
    h0, w0 = img0.shape[:2]

    # letterbox to IMG_SIZE
    img = letterbox(img0, new_shape=IMG_SIZE, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB->CHW
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device).float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)

    with torch.no_grad():
        pred = yolo_model(im)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)


    dets = []
    if pred and pred[0] is not None and len(pred[0]):
        det = pred[0]
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det.tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append([x1, y1, x2, y2, float(conf), int(cls)])
    return dets

def resnet_classify_crop(img_bgr: np.ndarray, box, resnet_model, device):
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(crop_rgb)
    x = resnet_tf(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = resnet_model(x)
        probs = torch.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(probs).item())
        conf = float(probs[cls].item())
    return cls, conf

def draw_boxes(img_bgr, dets, names, hybrid_results=None):
    """
    hybrid_results: list aligned with dets => (resnet_cls, resnet_conf)
    """
    out = img_bgr.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2, yconf, ycls = d
        if hybrid_results is not None and hybrid_results[i] is not None:
            rcls, rconf = hybrid_results[i]
            label = f"YOLO:{names[ycls]} {yconf:.2f} | RESNET:{names[rcls]} {rconf:.2f}"
        else:
            label = f"{names[ycls]} {yconf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out

# ----------------------------
# UI
# ----------------------------
st.title("🚦 Traffic Sign Detection Website (YOLOv7 + ResNet50)")

colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Settings")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device_str = st.selectbox("Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0 if device_str=="cpu" else 1)

    conf = st.slider("YOLO confidence threshold", 0.05, 0.95, YOLO_CONF_THRES, 0.05)
    iou = st.slider("YOLO IoU threshold", 0.10, 0.95, YOLO_IOU_THRES, 0.05)
    use_hybrid = st.checkbox("Use ResNet50 to re-classify each detected sign (Hybrid)", value=True)

    st.caption("Tip: If your CPU is slow, turn off Hybrid or increase confidence threshold.")

# Load models
names = load_class_names(CLASS_NAMES_YAML)
yolo_model, yolo_device = load_yolo(YOLO_WEIGHTS, device_str)

resnet_model = None
resnet_device = None
if use_hybrid:
    resnet_model, resnet_device, resnet_classes = load_resnet(
        RESNET_WEIGHTS, num_classes=len(names), device_str=device_str
    )

tab1, tab2, tab3 = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Webcam"])

# ----------------------------
# Image tab
# ----------------------------
with tab1:
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = Image.open(up).convert("RGB")
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        dets = yolo_infer_bgr(img_bgr, yolo_model, yolo_device, conf, iou)

        hybrid = None
        if use_hybrid and len(dets):
            hybrid = []
            for d in dets:
                x1, y1, x2, y2, _, _ = d
                hybrid.append(resnet_classify_crop(img_bgr, (x1, y1, x2, y2), resnet_model, resnet_device))

        vis = draw_boxes(img_bgr, dets, names, hybrid_results=hybrid)
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Input", use_column_width=True)
        with c2:
            st.image(vis_rgb, caption=f"Output (detections: {len(dets)})", use_column_width=True)

        if len(dets):
            st.subheader("Detections")
            rows = []
            for i, d in enumerate(dets):
                x1,y1,x2,y2,yconf,ycls = d
                row = {
                    "bbox": [x1,y1,x2,y2],
                    "yolo_cls": ycls,
                    "yolo_name": names[ycls],
                    "yolo_conf": round(yconf, 4),
                }
                if use_hybrid and hybrid and hybrid[i] is not None:
                    rcls, rconf = hybrid[i]
                    row.update({
                        "resnet_cls": rcls,
                        "resnet_name": names[rcls],
                        "resnet_conf": round(rconf, 4),
                    })
                rows.append(row)
            st.json(rows)

# ----------------------------
# Video tab (upload video file)
# ----------------------------
with tab2:
    vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if vid is not None:
        tfile = "temp_video_input"
        with open(tfile, "wb") as f:
            f.write(vid.read())

        cap = cv2.VideoCapture(tfile)
        stframe = st.empty()

        stop = st.button("Stop video processing")
        while cap.isOpened():
            if stop:
                break
            ret, frame = cap.read()
            if not ret:
                break

            dets = yolo_infer_bgr(frame, yolo_model, yolo_device, conf, iou)
            hybrid = None
            if use_hybrid and len(dets):
                hybrid = []
                for d in dets:
                    x1, y1, x2, y2, _, _ = d
                    hybrid.append(resnet_classify_crop(frame, (x1, y1, x2, y2), resnet_model, resnet_device))

            vis = draw_boxes(frame, dets, names, hybrid_results=hybrid)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            stframe.image(vis_rgb, channels="RGB", use_column_width=True)

        cap.release()
        try:
            os.remove(tfile)
        except:
            pass

# ----------------------------
# Webcam tab (streamlit-webrtc)
# ----------------------------
with tab3:
    st.caption("If webcam doesn’t open in your browser, try running in Chrome and allow camera permission.")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_time = 0

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")

            # simple throttle (optional) to reduce CPU usage
            now = time.time()
            if now - self.last_time < 0.05:
                return img
            self.last_time = now

            dets = yolo_infer_bgr(img, yolo_model, yolo_device, conf, iou)
            hybrid = None
            if use_hybrid and len(dets):
                hybrid = []
                for d in dets:
                    x1, y1, x2, y2, _, _ = d
                    hybrid.append(resnet_classify_crop(img, (x1, y1, x2, y2), resnet_model, resnet_device))

            vis = draw_boxes(img, dets, names, hybrid_results=hybrid)
            return vis

    webrtc_streamer(
        key="traffic-sign-webcam",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
