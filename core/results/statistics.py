"""
Statistical analytics and graphical plotting module (Phase 10).
Computes summary metrics and generates speed-vs-time trajectory plots.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


def compute_summary_statistics(
    duration_sec: float,
    ego_speed_records: List[Dict[str, Any]],
    vehicle_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes overall summary statistics for a completed analysis run.
    """
    ego_speeds = [r["speed_kmh"] for r in ego_speed_records if r.get("speed_kmh") is not None]
    veh_speeds = [r["vehicle_speed"] for r in vehicle_records if r.get("vehicle_speed") is not None]
    confs = [r.get("confidence", 0.0) for r in vehicle_records if r.get("confidence") is not None]

    unique_vehicles = len(set(r["vehicle_id"] for r in vehicle_records if r.get("vehicle_id")))

    avg_ego = float(np.mean(ego_speeds)) if ego_speeds else 0.0
    max_ego = float(np.max(ego_speeds)) if ego_speeds else 0.0
    avg_veh = float(np.mean(veh_speeds)) if veh_speeds else 0.0
    max_veh = float(np.max(veh_speeds)) if veh_speeds else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "duration_seconds": round(duration_sec, 2),
        "total_unique_vehicles": unique_vehicles,
        "ego_vehicle": {
            "average_speed_kmh": round(avg_ego, 1),
            "max_speed_kmh": round(max_ego, 1),
            "valid_samples": len(ego_speeds),
        },
        "surrounding_vehicles": {
            "average_speed_kmh": round(avg_veh, 1),
            "max_speed_kmh": round(max_veh, 1),
            "valid_estimates": len(veh_speeds),
        },
        "confidence": {
            "mean_confidence": round(mean_conf, 3),
        },
    }


def generate_speed_plots(
    ego_records: List[Dict[str, Any]],
    vehicle_records: List[Dict[str, Any]],
    output_png_path: Union[str, Path],
) -> Optional[Path]:
    """
    Generates a speed-versus-time graph using Matplotlib and saves to PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        out_file = Path(output_png_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.patch.set_facecolor("#1e1e1e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#282828")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.grid(True, linestyle="--", alpha=0.3, color="#666666")

        # Top Plot: Ego Speed
        ego_times = [r["timestamp"] for r in ego_records]
        ego_spds = [r["speed_kmh"] if r.get("speed_kmh") is not None else np.nan for r in ego_records]

        ax1.plot(ego_times, ego_spds, color="#00ff88", linewidth=2.0, label="Ego Vehicle Speed (km/h)")
        ax1.set_ylabel("Speed (km/h)")
        ax1.set_title("Ego Vehicle Speed Profile")
        ax1.legend(facecolor="#333333", labelcolor="white")

        # Bottom Plot: Surrounding Vehicles
        # Group by vehicle_id
        by_vehicle: Dict[int, Tuple[List[float], List[float]]] = {}
        for r in vehicle_records:
            vid = r.get("vehicle_id")
            spd = r.get("vehicle_speed")
            if vid and spd is not None:
                if vid not in by_vehicle:
                    by_vehicle[vid] = ([], [])
                by_vehicle[vid][0].append(r["timestamp"])
                by_vehicle[vid][1].append(spd)

        colors = ["#ffaa00", "#00d4ff", "#ff0077", "#aaff00", "#ff00ee"]
        for idx, (vid, (v_times, v_spds)) in enumerate(by_vehicle.items()):
            c = colors[idx % len(colors)]
            ax2.plot(v_times, v_spds, marker="o", markersize=3, linestyle="-", color=c, label=f"Vehicle #{vid}")

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Speed (km/h)")
        ax2.set_title("Surrounding Vehicles Speed Profile")
        if by_vehicle:
            ax2.legend(facecolor="#333333", labelcolor="white", loc="upper right")

        plt.tight_layout()
        plt.savefig(out_file, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return out_file
    except Exception as e:
        # Graceful fallback if matplotlib display / font issues arise
        return None
