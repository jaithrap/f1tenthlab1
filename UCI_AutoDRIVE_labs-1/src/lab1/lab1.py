#!/usr/bin/env python

# Import libraries
import socketio
import eventlet
from flask import Flask
import autodrive
import numpy as np
from typing import Optional

################################################################################

LEFT_WALL_TARGET_M = 0.45
LOOKAHEAD_M = 1
PID_KP, PID_KI, PID_KD = 0.8, 0.0, 0.02

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def build_pid(Kp: float, Ki: float, Kd: float, umin: float = -1.0, umax: float = 1.0, i_limit: float = 0.5):
    I = 0.0
    e_prev: Optional[float] = None
    def step(e: float, dt: Optional[float]) -> float:
        nonlocal I, e_prev
        if dt and dt > 0.0:
            I = clamp(I + e*dt, -i_limit, i_limit)
            d = 0.0 if e_prev is None else (e - e_prev)/dt
        else:
            d = 0.0
        e_prev = e
        u = Kp*e + Ki*I + Kd*d
        return clamp(u, umin, umax)
    return step

def get_range(scan: Optional[np.ndarray], angle_rad: float, min_angle: float = -3.0*np.pi/4.0, max_angle: float =  3.0*np.pi/4.0) -> float:
    if scan is None or not hasattr(scan, "size") or scan.size == 0:
        return 0.0
    a = float(np.clip(angle_rad, min_angle, max_angle))
    inc = (max_angle - min_angle) / max(scan.size - 1, 1)
    idx = int(round((a - min_angle) / inc))
    idx = max(0, min(idx, scan.size - 1))
    v = float(scan[idx])
    if not np.isfinite(v) or v <= 0.0:
        return 0.0
    return v

def left_wall_distance_and_alpha(scan: Optional[np.ndarray], theta_deg: float = 20.0):
    if scan is None:
        return None, None
    theta = np.deg2rad(theta_deg)
    a_ang_deg = 90.0 
    b_ang_deg = 90.0 + theta_deg
    a = get_range(scan, np.deg2rad(a_ang_deg))
    b = get_range(scan, np.deg2rad(b_ang_deg))
    print(f"LiDAR rays (deg): a={a_ang_deg:.1f}, b={b_ang_deg:.1f}")
    print(f"Ray distances (m): a={a:.3f}, b={b:.3f}")
    if a <= 0.0 or b <= 0.0:
        return None, None
    st = np.sin(theta)
    if abs(st) < 1e-6:
        return None, None
    alpha = np.arctan2(a*np.cos(theta) - b, a*st)
    D = b*np.cos(alpha)
    if not np.isfinite(D) or D <= 0.0:
        return None, None
    print(f"Left wall distance (m): {D:.3f}")
    return float(D), float(alpha)

# Initialize vehicle(s)
f1tenth_1 = autodrive.F1TENTH()
f1tenth_1.id = "V1"
steer_pid = build_pid(PID_KP, PID_KI, PID_KD)

# Initialize the server
sio = socketio.Server()
app = Flask(__name__)  # '__main__'


@sio.on("connect")
def connect(sid, environ):
    print("Connected!")


@sio.on("Bridge")
def bridge(sid, data):
    if data:
        f1tenth_1.parse_data(data)
        f1tenth_1.throttle_command = 0.2
        ranges = getattr(f1tenth_1, "lidar_range_array", None)
        dt = getattr(f1tenth_1, "delta_time", None) or 0.02

        D, alpha = left_wall_distance_and_alpha(ranges)
        if D is not None:
            alpha_s = alpha if (alpha is not None and np.isfinite(alpha)) else 0.0
            D_pred = D + LOOKAHEAD_M * np.sin(alpha_s)
            e = LEFT_WALL_TARGET_M - D_pred
            u = steer_pid(e, dt)
            steer = -u
            print(f"Distance={D:.3f}, Predicted={D_pred:.3f}, Error={e:.3f}, Steer={steer:.3f}")
        else:
            steer = 0.0

        f1tenth_1.steering_command = float(clamp(steer, -1.0, 1.0))

        json_msg = f1tenth_1.generate_commands()
        try:
            sio.emit("Bridge", data=json_msg)
        except Exception as exception_instance:
            print(exception_instance)

################################################################################

if __name__ == "__main__":
    app = socketio.Middleware(
        sio, app
    )  # Wrap flask application with socketio's middleware
    eventlet.wsgi.server(
        eventlet.listen(("", 4567)), app
    )  # Deploy as an eventlet WSGI server

