import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# -----------------------
# 0) Config (keep consistent with training)
# -----------------------
@dataclass
class CFG:
    IMG_SIZE: int = 336
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    AMP: bool = True


cfg = CFG()


# -----------------------
# 1) Preprocess: keep-aspect resize + pad to 336
#    and record meta for inverse mapping
# -----------------------
def preprocess_keep_aspect_pad(
    img: Image.Image,
    out_size: int = 336
) -> Tuple[torch.Tensor, Dict]:
    """
    Return:
      img_t: [3, out_size, out_size] float tensor in [0,1]
      meta: scale/pad/original size for inverse mapping
    """
    img = img.convert("RGB")
    w, h = img.size

    scale = min(out_size / w, out_size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    pad_w = out_size - new_w
    pad_h = out_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    img_t = F.to_tensor(img_resized)
    img_t = F.pad(img_t, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

    meta = {
        "orig_w": w,
        "orig_h": h,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "new_w": new_w,
        "new_h": new_h,
        "out_size": out_size,
    }
    return img_t, meta


def map_boxes_to_original(boxes_xyxy_336: np.ndarray, meta: Dict) -> np.ndarray:
    """
    boxes in 336-space -> original image space
    """
    scale = float(meta["scale"])
    pad_left = float(meta["pad_left"])
    pad_top = float(meta["pad_top"])
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    b = boxes_xyxy_336.astype(np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_left) / max(scale, 1e-8)
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_top) / max(scale, 1e-8)

    # clip
    b[:, 0] = np.clip(b[:, 0], 0, orig_w - 1)
    b[:, 1] = np.clip(b[:, 1], 0, orig_h - 1)
    b[:, 2] = np.clip(b[:, 2], 0, orig_w - 1)
    b[:, 3] = np.clip(b[:, 3], 0, orig_h - 1)

    # keep valid
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep]


# -----------------------
# 2) Model: identical to training
# -----------------------
class MultiTaskFasterRCNN(nn.Module):
    def __init__(self, img_size: int = 336):
        super().__init__()

        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

        # num_classes = 2 (background + fracture)
        in_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        # fixed 336x336 input
        detector.transform.min_size = (img_size,)
        detector.transform.max_size = img_size

        # same "16GB-friendly tweaks" as training
        detector.rpn.pre_nms_top_n_train = 1000
        detector.rpn.post_nms_top_n_train = 1000
        detector.rpn.pre_nms_top_n_test = 1000
        detector.rpn.post_nms_top_n_test = 300
        detector.roi_heads.batch_size_per_image = 128
        detector.roi_heads.detections_per_img = 100

        self.detector = detector

        # Global classifier head on FPN "pool" feature (C=256)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

    def forward(self, images: List[torch.Tensor]):
        # detector outputs (boxes/scores/labels)
        det_out = self.detector(images)

        # global cls
        imgs_t, _ = self.detector.transform(images, None)
        feats = self.detector.backbone(imgs_t.tensors)
        feat = feats["pool"] if ("pool" in feats) else list(feats.values())[-1]
        logits = self.cls_head(feat)  # [B,1]
        probs = torch.sigmoid(logits).squeeze(1)  # [B]
        return det_out, probs


# -----------------------
# 3) Visualization
# -----------------------
def draw_boxes_on_image(
    img: Image.Image,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    score_thr: float = 0.3
) -> Image.Image:
    """
    Draw on original image (PIL).
    """
    img_out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img_out)

    # font: try load default; Windows/Linux fallback
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), s in zip(boxes_xyxy, scores):
        if float(s) < score_thr:
            continue

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

        text = f"{float(s):.3f}"
        tx, ty = x1, max(0, y1 - 18)
        # background for text
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=(255, 0, 0))
        draw.text((tx + 3, ty + 2), text, fill=(255, 255, 255), font=font)

    return img_out


# -----------------------
# 4) Inference entry
# -----------------------
@torch.no_grad()
def run_inference(
    ckpt_path: str,
    image_path: str,
    out_path: str,
    score_thr: float = 0.3,
    save_json: Optional[str] = None
):
    device = cfg.DEVICE
    use_amp = cfg.AMP and device.startswith("cuda")

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # build model (same arch)
    model = MultiTaskFasterRCNN(img_size=cfg.IMG_SIZE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print("[WARN] load_state_dict not strict.")
        print("  missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        print("  unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.to(device)
    model.eval()

    # load image
    img_orig = Image.open(image_path).convert("RGB")
    img_t, meta = preprocess_keep_aspect_pad(img_orig, out_size=cfg.IMG_SIZE)

    images = [img_t.to(device)]

    # forward
    with torch.cuda.amp.autocast(enabled=use_amp):
        det_out, global_probs = model(images)

    pred = det_out[0]
    global_prob = float(global_probs[0].detach().cpu().item())

    # extract boxes/scores (in 336-space)
    boxes_336 = pred["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
    labels = pred.get("labels", torch.zeros((len(scores),), dtype=torch.int64)).detach().cpu().numpy()

    # map to original coords
    boxes_orig = map_boxes_to_original(boxes_336, meta)

    # Note: mapping may drop invalid boxes; keep aligned with scores by re-filtering with same mask
    # We will compute keep mask again on original mapping by checking if the inverse kept it.
    # Simpler: rebuild keep based on original-mapped size by repeating mapping validity on 336 first.
    # Here we do it robustly:
    mapped = []
    mapped_scores = []
    mapped_labels = []
    for b, s, lab in zip(boxes_336, scores, labels):
        bb = map_boxes_to_original(b[None, :], meta)
        if bb.shape[0] == 0:
            continue
        mapped.append(bb[0])
        mapped_scores.append(float(s))
        mapped_labels.append(int(lab))
    if len(mapped) == 0:
        mapped = np.zeros((0, 4), dtype=np.float32)
        mapped_scores = np.zeros((0,), dtype=np.float32)
        mapped_labels = np.zeros((0,), dtype=np.int32)
    else:
        mapped = np.stack(mapped, axis=0).astype(np.float32)
        mapped_scores = np.array(mapped_scores, dtype=np.float32)
        mapped_labels = np.array(mapped_labels, dtype=np.int32)

    # draw
    vis = draw_boxes_on_image(img_orig, mapped, mapped_scores, score_thr=score_thr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vis.save(out_path)

    # optional json
    result = {
        "image_path": image_path,
        "out_path": out_path,
        "global_fracture_prob": global_prob,
        "score_thr": float(score_thr),
        "detections": [
            {
                "bbox_xyxy": [float(x) for x in b.tolist()],
                "score": float(s),
                "label": int(l),
            }
            for b, s, l in zip(mapped, mapped_scores, mapped_labels)
            if float(s) >= score_thr
        ],
    }

    if save_json is not None:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    # print summary
    print("==== Inference Result ====")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Image:      {image_path}")
    print(f"Output:     {out_path}")
    print(f"Global fracture prob (global cls): {global_prob:.4f}")
    print(f"Num dets (score >= {score_thr}): {len(result['detections'])}")
    for i, d in enumerate(result["detections"][:10]):
        print(f"  #{i}: score={d['score']:.3f}, bbox={d['bbox_xyxy']}")
    if len(result["detections"]) > 10:
        print("  ...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="best_multitask_frcnn_verbose2.pt",
                        help="Path to best_multitask_frcnn_verbose2.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to an X-ray image")
    parser.add_argument("--out", type=str, default="out_with_boxes.jpg", help="Output image path")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--json", type=str, default=None, help="Optional output json path")
    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        image_path=args.image,
        out_path=args.out,
        score_thr=args.score_thr,
        save_json=args.json
    )


if __name__ == "__main__":
    main()


