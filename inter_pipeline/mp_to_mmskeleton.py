"""
mp_to_mmskeleton.py
===================

Glue between the two repos in this workspace:

    HumanPoseAction  (MediaPipe BlazePose, 33 keypoints)  ->  mmskeleton (ST-GCN)

HumanPoseAction's `pose_detection.py --mode body` exports a JSON of the form:

    {
      "metadata": { "processed_frame_w": W, "processed_frame_h": H,
                    "source_file": "...", "source_fps": 30.0, ... },
      "frames": [ { "frame_id": 0, "detected": true,
                    "keypoints": { "NOSE": {"x_norm":.., "y_norm":.., "visibility":..}, ... } },
                  ... ]
    }

mmskeleton's `datasets.SkeletonLoader` (mmskeleton/datasets/skeleton/loader.py) expects
ONE JSON per clip of the form:

    {
      "info": { "video_name": str, "resolution": [W, H], "num_frame": T,
                "num_keypoints": V, "keypoint_channels": ["x", "y", "score"] },
      "annotations": [ { "frame_index": t, "id": 0, "person_id": 0,
                         "keypoints": [[x, y, score], ... V rows ] }, ... ],
      "category_id": int
    }

MediaPipe's 33 landmarks are a SUPERSET of every ST-GCN layout, so the conversion is a
pure re-indexing (plus a couple of derived joints such as OpenPose's "neck" =
mid-shoulder).  This module supports the three layouts defined in
`mmskeleton/ops/st_gcn/graph.py`:

    layout="openpose"     18 joints  -> matches the pretrained  st_gcn.kinetics  checkpoint
    layout="coco"         17 joints  -> matches configs/recognition/st_gcn/dataset_example
    layout="ntu-rgb+d"    25 joints  -> matches the pretrained  st_gcn.ntu-*     checkpoints

Run standalone:

    python mp_to_mmskeleton.py input_hpa.json output_skeleton.json --layout openpose --category 57
"""

import argparse
import json
import os


# --------------------------------------------------------------------------------------
# Layout definitions.  Each entry is (target_joint_index -> recipe).
# A recipe is either:
#   a MediaPipe landmark NAME (str)                      -> copy that landmark
#   a tuple of names                                     -> average them (midpoint / proxy)
# MediaPipe BlazePose landmark names are exactly the keys HumanPoseAction writes.
# --------------------------------------------------------------------------------------

# OpenPose BODY_18 order used by ST-GCN's "openpose" graph (kinetics checkpoint).
OPENPOSE_18 = [
    "NOSE",                                 # 0  Nose
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),    # 1  Neck (mid-shoulder)
    "RIGHT_SHOULDER",                       # 2  RShoulder
    "RIGHT_ELBOW",                          # 3  RElbow
    "RIGHT_WRIST",                          # 4  RWrist
    "LEFT_SHOULDER",                        # 5  LShoulder
    "LEFT_ELBOW",                           # 6  LElbow
    "LEFT_WRIST",                           # 7  LWrist
    "RIGHT_HIP",                            # 8  RHip
    "RIGHT_KNEE",                           # 9  RKnee
    "RIGHT_ANKLE",                          # 10 RAnkle
    "LEFT_HIP",                             # 11 LHip
    "LEFT_KNEE",                            # 12 LKnee
    "LEFT_ANKLE",                           # 13 LAnkle
    "RIGHT_EYE",                            # 14 REye
    "LEFT_EYE",                             # 15 LEye
    "RIGHT_EAR",                            # 16 REar
    "LEFT_EAR",                             # 17 LEar
]

# COCO-17 order used by ST-GCN's "coco" graph (dataset_example / HRNet path).
COCO_17 = [
    "NOSE",             # 0
    "LEFT_EYE",         # 1
    "RIGHT_EYE",        # 2
    "LEFT_EAR",         # 3
    "RIGHT_EAR",        # 4
    "LEFT_SHOULDER",    # 5
    "RIGHT_SHOULDER",   # 6
    "LEFT_ELBOW",       # 7
    "RIGHT_ELBOW",      # 8
    "LEFT_WRIST",       # 9
    "RIGHT_WRIST",      # 10
    "LEFT_HIP",         # 11
    "RIGHT_HIP",        # 12
    "LEFT_KNEE",        # 13
    "RIGHT_KNEE",       # 14
    "LEFT_ANKLE",       # 15
    "RIGHT_ANKLE",      # 16
]

# NTU RGB+D / Kinect-v2 25-joint order used by ST-GCN's "ntu-rgb+d" graph
# (st_gcn.ntu-xsub / ntu-xview checkpoints).  Several NTU joints have no direct
# MediaPipe equivalent and are derived as midpoints / nearest proxies.
NTU_25 = [
    ("LEFT_HIP", "RIGHT_HIP"),                                  # 0  SpineBase
    ("LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER"),  # 1 SpineMid
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),                       # 2  Neck
    ("LEFT_EAR", "RIGHT_EAR"),                                 # 3  Head
    "LEFT_SHOULDER",                                           # 4  ShoulderLeft
    "LEFT_ELBOW",                                              # 5  ElbowLeft
    "LEFT_WRIST",                                              # 6  WristLeft
    ("LEFT_INDEX", "LEFT_PINKY"),                             # 7  HandLeft (palm)
    "RIGHT_SHOULDER",                                          # 8  ShoulderRight
    "RIGHT_ELBOW",                                             # 9  ElbowRight
    "RIGHT_WRIST",                                             # 10 WristRight
    ("RIGHT_INDEX", "RIGHT_PINKY"),                          # 11 HandRight (palm)
    "LEFT_HIP",                                                # 12 HipLeft
    "LEFT_KNEE",                                               # 13 KneeLeft
    "LEFT_ANKLE",                                              # 14 AnkleLeft
    "LEFT_FOOT_INDEX",                                         # 15 FootLeft
    "RIGHT_HIP",                                               # 16 HipRight
    "RIGHT_KNEE",                                              # 17 KneeRight
    "RIGHT_ANKLE",                                             # 18 AnkleRight
    "RIGHT_FOOT_INDEX",                                        # 19 FootRight
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),                      # 20 SpineShoulder
    "LEFT_INDEX",                                              # 21 HandTipLeft
    "LEFT_THUMB",                                              # 22 ThumbLeft
    "RIGHT_INDEX",                                             # 23 HandTipRight
    "RIGHT_THUMB",                                             # 24 ThumbRight
]

LAYOUTS = {
    "openpose": OPENPOSE_18,
    "coco": COCO_17,
    "ntu-rgb+d": NTU_25,
}


def _landmark_xys(kp_dict, name, use_pixels, width, height):
    """Return (x, y, score) for one MediaPipe landmark, or None if absent.

    Coordinates are emitted in the [0, 1] normalised space that every ST-GCN
    preprocessing step (mmskeleton `normalize_by_resolution`) expects.  If the
    HumanPoseAction JSON only carries pixel coordinates we divide by the frame
    size; otherwise we use the ready-made *_norm fields.
    """
    lm = kp_dict.get(name)
    if lm is None:
        return None
    score = float(lm.get("visibility", 1.0))
    if use_pixels:
        x = float(lm["x_px"]) / width
        y = float(lm["y_px"]) / height
    else:
        x = float(lm["x_norm"])
        y = float(lm["y_norm"])
    return (x, y, score)


def _resolve_joint(kp_dict, recipe, use_pixels, width, height):
    """Resolve one target joint (direct copy or averaged proxy) -> [x, y, score]."""
    names = (recipe,) if isinstance(recipe, str) else recipe
    pts = [_landmark_xys(kp_dict, n, use_pixels, width, height) for n in names]
    pts = [p for p in pts if p is not None and p[2] > 0.0]
    if not pts:
        return [0.0, 0.0, 0.0]  # missing joint -> zero (masked out downstream)
    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    score = sum(p[2] for p in pts) / len(pts)
    return [round(x, 6), round(y, 6), round(score, 6)]


def convert(hpa_json, layout="openpose", category_id=-1, use_pixels=False):
    """Convert a loaded HumanPoseAction dict -> mmskeleton skeleton dict."""
    if layout not in LAYOUTS:
        raise ValueError(
            "Unknown layout %r. Choose one of %s" % (layout, list(LAYOUTS)))
    joint_map = LAYOUTS[layout]
    num_keypoints = len(joint_map)

    meta = hpa_json.get("metadata", {})
    width = meta.get("processed_frame_w") or meta.get("original_frame_w") or 1
    height = meta.get("processed_frame_h") or meta.get("original_frame_h") or 1
    frames = hpa_json.get("frames", [])
    num_frame = len(frames)

    annotations = []
    for frame in frames:
        t = frame.get("frame_id", len(annotations))
        if not frame.get("detected") or not frame.get("keypoints"):
            continue  # SkeletonLoader leaves absent frames as zeros
        kp_dict = frame["keypoints"]
        keypoints = [
            _resolve_joint(kp_dict, recipe, use_pixels, width, height)
            for recipe in joint_map
        ]
        annotations.append({
            "frame_index": int(t),
            "id": 0,
            "person_id": 0,
            "keypoints": keypoints,
        })

    # We emit already-normalised [0,1] coords, so resolution=[1,1] makes
    # mmskeleton's normalize_by_resolution a pure (x - 0.5) centering.
    return {
        "info": {
            "video_name": meta.get("source_file", "unknown"),
            "resolution": [1, 1],
            "num_frame": num_frame,
            "num_keypoints": num_keypoints,
            "keypoint_channels": ["x", "y", "score"],
            "layout": layout,
            "source": meta.get("source"),
            "fps": meta.get("source_fps"),
        },
        "annotations": annotations,
        "category_id": int(category_id),
    }


def convert_file(in_path, out_path, layout="openpose", category_id=-1,
                 use_pixels=False):
    with open(in_path, "r", encoding="utf-8") as f:
        hpa = json.load(f)
    result = convert(hpa, layout=layout, category_id=category_id,
                     use_pixels=use_pixels)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


def _cli():
    p = argparse.ArgumentParser(
        description="Convert HumanPoseAction keypoint JSON to mmskeleton "
                    "SkeletonLoader JSON.")
    p.add_argument("input", help="HumanPoseAction *_keypoints_*.json")
    p.add_argument("output", help="destination mmskeleton skeleton JSON")
    p.add_argument("--layout", default="openpose", choices=list(LAYOUTS),
                   help="target ST-GCN joint layout (default: openpose)")
    p.add_argument("--category", type=int, default=-1,
                   help="class label id for this clip (-1 = unknown/inference)")
    p.add_argument("--use-pixels", action="store_true",
                   help="use x_px/y_px instead of x_norm/y_norm")
    args = p.parse_args()

    res = convert_file(args.input, args.output, layout=args.layout,
                       category_id=args.category, use_pixels=args.use_pixels)
    info = res["info"]
    print("[mp_to_mmskeleton] layout=%s  V=%d  frames_total=%d  frames_detected=%d"
          % (info["layout"], info["num_keypoints"], info["num_frame"],
             len(res["annotations"])))
    print("[mp_to_mmskeleton] wrote %s" % args.output)


if __name__ == "__main__":
    _cli()
