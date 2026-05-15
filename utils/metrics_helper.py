import math

def dist2d(a, b):
    return math.sqrt((a["x_px"] - b["x_px"])**2 + (a["y_px"] - b["y_px"])**2)

def velocity(kp_prev, kp_curr, dt):
    """Viteza în pixeli/secundă între două frame-uri."""
    if kp_prev is None or kp_curr is None:
        return 0.0
    dx = kp_curr["x_px"] - kp_prev["x_px"]
    dy = kp_curr["y_px"] - kp_prev["y_px"]
    return math.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0.0
