"""
build_dataset.py
================

Batch-convert a folder of HumanPoseAction exports into an mmskeleton skeleton
dataset directory that `datasets.SkeletonLoader` can consume for TRAINING /
TESTING an ST-GCN from scratch on your own MediaPipe data.

This is the accurate route (as opposed to zero-shot inference with a pretrained
checkpoint): you train ST-GCN directly on MediaPipe keypoints, so there is no
cross-domain gap.  It is the path wired into
`configs/recognition/st_gcn/humanpose_coco/{train,test}.yaml`.

Labels can come from either:
  * a category-annotation JSON (same shape as mmskeleton's
    resource/category_annotation_example.json)::

        { "categories": ["clap", "wave", ...],
          "annotations": { "S001..._keypoints.json": {"category_id": 0}, ... } }

  * NTU RGB+D filenames: pass --from-ntu-filename to read the "A0xx" action id
    out of names like  S001C001P001R001A010_rgb_keypoints_full_body.json
    (category_id = xx - 1, matching pipeline/labels.py:NTU_60).

Example
-------
    python build_dataset.py ../HumanPoseAction/output ./data/my_dataset \
        --layout coco --from-ntu-filename --val-split 0.25
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

import HumanPoseAction.inter_pipeline.models.mp_to_mmskeleton as mp_to_mmskeleton

NTU_ACTION_RE = re.compile(r"A(\d{3})")
NTU_SUBJECT_RE = re.compile(r"P(\d{3})")


def ntu_category_from_name(filename):
    """Return 0-indexed NTU action id from an S..A0xx.. filename, or None."""
    m = NTU_ACTION_RE.search(os.path.basename(filename))
    if not m:
        return None
    return int(m.group(1)) - 1  # A010 -> 9


def ntu_subject_from_name(filename):
    """Return the P0xx subject id from an NTU filename, or None."""
    m = NTU_SUBJECT_RE.search(os.path.basename(filename))
    return m.group(1) if m else None


def split_for_subject(subject, val_split, seed):
    """Deterministic train/val assignment for one subject.

    Hashing the subject id (rather than drawing from a sequential RNG per
    file) gives two properties a per-clip random split doesn't:
      - every clip from the same subject lands in the same split, so
        near-duplicate clips (same person/take, different camera or
        repetition) can never straddle the train/val boundary and inflate
        val accuracy with memorized near-duplicates -- the same reason NTU's
        own X-Sub benchmark holds out whole subjects, not random clips.
      - the assignment doesn't depend on how many files exist or the order
        they're processed in, so it stays stable as more clips get added
        (e.g. while extraction is still running) instead of reshuffling
        previously-built train/val membership on every rebuild.
    """
    if val_split <= 0:
        return "train"
    digest = hashlib.md5(f"{seed}:{subject}".encode()).hexdigest()
    frac = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if frac < val_split else "train"


def load_category_annotation(path):
    with open(path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    categories = ann.get("categories", [])
    mapping = {k: v["category_id"] for k, v in ann.get("annotations", {}).items()}
    return categories, mapping


def build(input_dir, output_dir, layout="coco", annotation=None,
          from_ntu=False, val_split=0.0, seed=0, only_actions=None):
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not files:
        raise SystemExit("[build] no *.json found in %s" % input_dir)

    categories, ann_map = ([], {})
    if annotation:
        categories, ann_map = load_category_annotation(annotation)

    # only_actions is a set of 0-based category ids to keep. When set, kept
    # classes are remapped to a contiguous 0..N-1 range so the trained model
    # has exactly N outputs (not max_id+1 with empty slots) and the random
    # baseline is a clean 1/N. remap[original_id] = new_id, in sorted order.
    remap = None
    if only_actions is not None:
        remap = {orig: new for new, orig in enumerate(sorted(only_actions))}

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    if val_split > 0:
        os.makedirs(val_dir, exist_ok=True)

    rng = random.Random(seed)
    seen_categories = set()
    written = {"train": 0, "val": 0}
    skipped = []

    for path in files:
        name = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            probe = json.load(f)
        if "frames" not in probe or "metadata" not in probe:
            continue  # not a HumanPoseAction keypoints export (e.g. *_metrics.json)
        if name in ann_map:
            cat = ann_map[name]
        elif from_ntu:
            cat = ntu_category_from_name(name)
        else:
            cat = -1
        if cat is None or cat < 0:
            skipped.append(name)
            continue
        if remap is not None:
            if cat not in remap:
                continue  # action not in the requested subset -- silently drop
            cat = remap[cat]
        seen_categories.add(cat)

        # Split by subject when possible (see split_for_subject) so clips of
        # the same person/take never straddle train/val; per-clip random is
        # the only option left for annotation-based labeling, which has no
        # subject concept.
        subject = ntu_subject_from_name(name) if from_ntu else None
        if subject is not None:
            split = split_for_subject(subject, val_split, seed)
        else:
            split = "val" if (val_split > 0 and rng.random() < val_split) else "train"
        out_name = os.path.splitext(name)[0] + "__%s.json" % layout
        out_path = os.path.join(train_dir if split == "train" else val_dir, out_name)
        mp_to_mmskeleton.convert_file(path, out_path, layout=layout, category_id=cat)
        written[split] += 1

    # Record the remapped-id -> NTU-action-name mapping so results are readable.
    if remap is not None:
        try:
            import HumanPoseAction.inter_pipeline.models.labels as label_maps
            label_names = label_maps.NTU_60
        except Exception:
            label_names = None
        label_map = {
            new: {"ntu_action_id": orig + 1,
                  "label": (label_names[orig] if label_names and orig < len(label_names)
                            else "action_%d" % (orig + 1))}
            for orig, new in remap.items()
        }
        with open(os.path.join(output_dir, "label_map.json"), "w",
                  encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

    print("[build] layout=%s  train=%d  val=%d  skipped(no label)=%d"
          % (layout, written["train"], written["val"], len(skipped)))
    print("[build] distinct category ids: %s" % sorted(seen_categories))
    if remap is not None:
        print("[build] class remap (new_id <- NTU A0xx): %s"
              % {new: "A%03d" % (orig + 1) for orig, new in remap.items()})
    if skipped:
        print("[build] skipped: %s%s" % (skipped[:5], " ..." if len(skipped) > 5 else ""))
    print("[build] dataset written to %s" % os.path.abspath(output_dir))
    print("[build] -> num_class = %d" % (max(seen_categories) + 1 if seen_categories else 0))
    return written


def _cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", help="folder of HumanPoseAction *_keypoints_*.json")
    p.add_argument("output_dir", help="destination dataset dir (train/ + val/)")
    p.add_argument("--layout", default="coco",
                   choices=list(mp_to_mmskeleton.LAYOUTS))
    p.add_argument("--annotation", help="category-annotation JSON (filename->id)")
    p.add_argument("--from-ntu-filename", action="store_true",
                   help="derive labels from A0xx in NTU filenames")
    p.add_argument("--val-split", type=float, default=0.0,
                   help="fraction of clips to route to val/ (default 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-actions", type=int, nargs="+", metavar="A",
                   help="keep only these NTU action numbers (1-based, e.g. "
                        "8 9 23 27 43 for sit/stand/wave/jump/fall); kept "
                        "classes are remapped to a contiguous 0..N-1 range")
    args = p.parse_args()
    # CLI takes 1-based NTU action numbers (A008 -> 8); convert to the 0-based
    # category ids used internally (matching ntu_category_from_name).
    only_actions = ({a - 1 for a in args.only_actions}
                    if args.only_actions else None)
    build(args.input_dir, args.output_dir, layout=args.layout,
          annotation=args.annotation, from_ntu=args.from_ntu_filename,
          val_split=args.val_split, seed=args.seed, only_actions=only_actions)


if __name__ == "__main__":
    _cli()
