# ── Body metrics ───────────────────────────────────────────────────────────────
_COM_LANDMARKS = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]

def center_of_mass(kps):
    """Average position of available torso landmarks (shoulders + hips)."""
    pts = [kps[k] for k in _COM_LANDMARKS if k in kps]
    if not pts:
        return None
    return {
        "x_px": sum(p["x_px"] for p in pts) / len(pts),
        "y_px": sum(p["y_px"] for p in pts) / len(pts),
    }