"""
Data Simulation Module
======================
Simulates real-time crowd sensor data that would come from mmWave radar sensors
(LD2410 + ESP32) in a real deployment. Generates time-series data for 3 zones
with realistic congestion patterns.

Each data point contains:
  - zone_id:   Which zone the sensor is monitoring (A, B, C)
  - timestamp: When the reading was taken
  - density:   People per square meter (0-10 scale)
  - velocity:  Average crowd movement speed in m/s (0-2 scale)

Congestion physics:
  - Normal: low density (0.5-2.0), high velocity (1.0-1.8)
  - Building: density rising (2.0-5.0), velocity dropping (0.5-1.0)
  - Congested: high density (5.0-9.0), very low velocity (0.05-0.3)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def _add_noise(signal: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add Gaussian noise and random spikes to simulate real sensor imperfections."""
    noise = np.random.normal(0, noise_level, len(signal))
    
    # Add random positive spikes (e.g. sensor ghosting or sudden clustering)
    spike_mask = np.random.rand(len(signal)) < 0.05  # 5% chance of spike
    spikes = np.random.exponential(scale=noise_level * 10, size=len(signal)) * spike_mask
    
    return signal + noise + spikes


def _generate_zone_data(
    zone_id: str,
    start_time: datetime,
    n_points: int,
    interval_seconds: int,
    density_profile: np.ndarray,
    velocity_profile: np.ndarray,
    noise_level: float = 0.15,
) -> pd.DataFrame:
    """
    Generate time-series data for a single zone given density/velocity profiles.
    Adds realistic sensor noise and clamps values to physical bounds.
    """
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(n_points)]

    density = _add_noise(density_profile, noise_level)
    velocity = _add_noise(velocity_profile, noise_level * 0.5)

    # Clamp to realistic physical bounds
    density = np.clip(density, 0.1, 10.0)
    velocity = np.clip(velocity, 0.02, 2.0)

    return pd.DataFrame({
        "zone_id": zone_id,
        "timestamp": timestamps,
        "density": np.round(density, 3),
        "velocity": np.round(velocity, 3),
    })


def _build_congestion_event(
    n_before: int, n_ramp: int, n_peak: int, n_recovery: int,
    base_density: float = 1.0, peak_density: float = 7.5,
    base_velocity: float = 1.5, min_velocity: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a single congestion event profile:
      1. Normal period    → low density, high velocity
      2. Ramp-up period   → density increases, velocity decreases
      3. Peak congestion  → high density, low velocity
      4. Recovery period  → density decreases, velocity recovers
    """
    # 1. Normal
    d_normal = np.full(n_before, base_density)
    v_normal = np.full(n_before, base_velocity)

    # 2. Ramp-up (the critical prediction window — 10-15 min before peak)
    d_ramp = np.linspace(base_density, peak_density, n_ramp)
    v_ramp = np.linspace(base_velocity, min_velocity, n_ramp)

    # 3. Peak congestion
    d_peak = np.full(n_peak, peak_density)
    v_peak = np.full(n_peak, min_velocity)

    # 4. Recovery
    d_recovery = np.linspace(peak_density, base_density * 1.5, n_recovery)
    v_recovery = np.linspace(min_velocity, base_velocity * 0.8, n_recovery)

    density = np.concatenate([d_normal, d_ramp, d_peak, d_recovery])
    velocity = np.concatenate([v_normal, v_ramp, v_peak, v_recovery])

    return density, velocity


def generate_normal_day(start_time: datetime = None, interval_seconds: int = 30) -> pd.DataFrame:
    """
    Scenario 1: Normal Day
    - Moderate crowd levels with 2 small congestion events (e.g., lunch rush, end of day)
    - Each zone behaves slightly differently
    """
    if start_time is None:
        start_time = datetime(2026, 2, 25, 8, 0, 0)

    frames = []

    # Zone A — main entrance, 2 mild congestion events
    d1, v1 = _build_congestion_event(60, 30, 15, 30, 1.0, 5.5, 1.5, 0.3)
    d2, v2 = _build_congestion_event(40, 25, 10, 25, 1.2, 6.0, 1.4, 0.25)
    d_a = np.concatenate([d1, d2, np.full(45, 1.0)])
    v_a = np.concatenate([v1, v2, np.full(45, 1.5)])
    frames.append(_generate_zone_data("Zone_A", start_time, len(d_a), interval_seconds, d_a, v_a))

    # Zone B — side corridor, 1 congestion event
    n_total = len(d_a)
    d_b_base = np.full(n_total, 0.8)
    v_b_base = np.full(n_total, 1.6)
    # Insert one congestion event in the middle
    ev_start = 100
    d_ev, v_ev = _build_congestion_event(0, 25, 12, 20, 0.8, 5.0, 1.6, 0.35)
    ev_len = len(d_ev)
    d_b_base[ev_start:ev_start + ev_len] = d_ev
    v_b_base[ev_start:ev_start + ev_len] = v_ev
    frames.append(_generate_zone_data("Zone_B", start_time, n_total, interval_seconds, d_b_base, v_b_base))

    # Zone C — open area, no congestion (control zone)
    d_c = np.full(n_total, 0.6) + np.sin(np.linspace(0, 4 * np.pi, n_total)) * 0.3
    v_c = np.full(n_total, 1.7) - np.sin(np.linspace(0, 4 * np.pi, n_total)) * 0.15
    frames.append(_generate_zone_data("Zone_C", start_time, n_total, interval_seconds, d_c, v_c))

    return pd.concat(frames, ignore_index=True)


def generate_post_event_rush(start_time: datetime = None, interval_seconds: int = 30) -> pd.DataFrame:
    """
    Scenario 2: Post-Event Rush
    - Sudden surge in all zones after a large event ends
    - All 3 zones experience congestion in a cascade pattern
    """
    if start_time is None:
        start_time = datetime(2026, 2, 25, 21, 0, 0)

    frames = []

    # Zone A — first to congest (closest to venue)
    d_a, v_a = _build_congestion_event(5, 25, 25, 40, 2.5, 8.5, 1.4, 0.08)
    padding = np.full(25, 1.0)
    d_a = np.concatenate([d_a, padding])
    v_a = np.concatenate([v_a, np.full(25, 1.5)])
    frames.append(_generate_zone_data("Zone_A", start_time, len(d_a), interval_seconds, d_a, v_a))

    n_total = len(d_a)

    # Zone B — cascading congestion (starts earlier now)
    d_b = np.full(n_total, 2.0)
    v_b = np.full(n_total, 1.5)
    d_ev, v_ev = _build_congestion_event(0, 25, 20, 35, 2.0, 7.5, 1.5, 0.12)
    ev_start = 15  # Much shorter delay
    ev_len = min(len(d_ev), n_total - ev_start)
    d_b[ev_start:ev_start + ev_len] = d_ev[:ev_len]
    v_b[ev_start:ev_start + ev_len] = v_ev[:ev_len]
    frames.append(_generate_zone_data("Zone_B", start_time, n_total, interval_seconds, d_b, v_b))

    # Zone C — latest cascade
    d_c = np.full(n_total, 1.5)
    v_c = np.full(n_total, 1.6)
    d_ev2, v_ev2 = _build_congestion_event(0, 20, 15, 30, 1.5, 6.0, 1.6, 0.15)
    ev_start2 = 25  # Shorter delay
    ev_len2 = min(len(d_ev2), n_total - ev_start2)
    d_c[ev_start2:ev_start2 + ev_len2] = d_ev2[:ev_len2]
    v_c[ev_start2:ev_start2 + ev_len2] = v_ev2[:ev_len2]
    frames.append(_generate_zone_data("Zone_C", start_time, n_total, interval_seconds, d_c, v_c))

    return pd.concat(frames, ignore_index=True)


def generate_emergency_evacuation(start_time: datetime = None, interval_seconds: int = 30) -> pd.DataFrame:
    """
    Scenario 3: Emergency Evacuation
    - Extremely rapid density spikes in all zones simultaneously
    - Velocity drops to near-zero (people stuck)
    - The model MUST catch this instantly
    """
    if start_time is None:
        start_time = datetime(2026, 2, 25, 14, 0, 0)

    frames = []

    # All zones spike almost simultaneously with very steep ramps, already elevated baseline
    for zone_id, (base_d, peak_d, base_v) in [
        ("Zone_A", (3.5, 9.8, 1.0)),
        ("Zone_B", (3.0, 9.5, 1.2)),
        ("Zone_C", (2.5, 9.0, 1.4)),
    ]:
        d_ev, v_ev = _build_congestion_event(
            n_before=2, n_ramp=8, n_peak=40, n_recovery=35,
            base_density=base_d, peak_density=peak_d,
            base_velocity=base_v, min_velocity=0.02,
        )
        frames.append(_generate_zone_data(
            zone_id, start_time, len(d_ev), interval_seconds, d_ev, v_ev, noise_level=0.3
        ))

    return pd.concat(frames, ignore_index=True)


def generate_training_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a large combined dataset for model training.
    Includes multiple runs of all 3 scenarios with different random seeds
    to give the model enough examples.
    """
    np.random.seed(seed)
    frames = []

    base_time = datetime(2026, 1, 1, 8, 0, 0)

    for i in range(5):
        np.random.seed(seed + i)
        t = base_time + timedelta(days=i * 3)
        frames.append(generate_normal_day(start_time=t))

    for i in range(5):
        np.random.seed(seed + 100 + i)
        t = base_time + timedelta(days=15 + i * 3)
        frames.append(generate_post_event_rush(start_time=t))

    for i in range(3):
        np.random.seed(seed + 200 + i)
        t = base_time + timedelta(days=30 + i * 3)
        frames.append(generate_emergency_evacuation(start_time=t))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    return combined


def stream_live_data(scenario_fn, interval_seconds: int = 30):
    """
    Generator that yields one row at a time to simulate a live data feed.
    Used by the Streamlit dashboard.
    """
    df = scenario_fn()
    zones = df["zone_id"].unique()
    n_per_zone = len(df[df["zone_id"] == zones[0]])

    for i in range(n_per_zone):
        rows = []
        for zone in zones:
            zone_df = df[df["zone_id"] == zone].reset_index(drop=True)
            if i < len(zone_df):
                rows.append(zone_df.iloc[i])
        yield pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    df = generate_training_dataset()
    print(f"Training dataset shape: {df.shape}")
    print(f"Zones: {df['zone_id'].unique()}")
    print(f"Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nSample data:\n{df.head(10)}")
    print(f"\nDensity stats:\n{df['density'].describe()}")
    print(f"\nVelocity stats:\n{df['velocity'].describe()}")
