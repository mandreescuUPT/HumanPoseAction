"""Action label maps for the pretrained ST-GCN checkpoints in mmskeleton/checkpoints.

- Kinetics-Skeleton (400 classes) matches  st_gcn.kinetics-*.pth   (layout "openpose").
- NTU RGB+D       (60 classes) matches  st_gcn.ntu-xsub-*.pth /
                                        st_gcn.ntu-xview-*.pth      (layout "ntu-rgb+d").
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_kinetics_labels():
    """400 Kinetics-Skeleton class names, index-aligned to the checkpoint fcn."""
    path = os.path.join(_HERE, "kinetics_label_name.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# NTU RGB+D 60 action classes, in official A001..A060 order (index i == class i,
# which is A{i+1:03d}).  The sample clip S001C001P001R001A010 is therefore label 9.
NTU_60 = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping (one foot jumping)",
    "jump up", "make a phone call/answer phone", "playing with phone/tablet",
    "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together",
    "nod head/bow", "shake head", "wipe face", "salute",
    "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other",
]


def labels_for(layout):
    if layout == "openpose":
        return load_kinetics_labels()
    if layout == "ntu-rgb+d":
        return NTU_60
    return None  # coco -> depends on the dataset you train on
