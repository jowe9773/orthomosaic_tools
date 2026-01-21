import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from math import ceil


def build_experiment_dict(filepath):
    """
    Load an annotation JSON file and organize point data into a dictionary
    of experiments keyed by date, where each value is a list of pandas
    DataFrames (one per camera).

    Parameters
    ----------
    filepath : str or Path
        Path to the annotation JSON file.

    Returns
    -------
    dict
        {
            "YYYYMMDD": [df_cam1, df_cam2, ...],
            ...
        }
    """

    filepath = Path(filepath)

    with open(filepath, "r") as f:
        data = json.load(f)

    # Nested structure:
    # experiments[date][camera_id] -> list of point dicts
    experiments = defaultdict(lambda: defaultdict(list))

    for image_entry in data:
        # ---- parse filename ----
        filename = Path(image_entry["file_upload"]).name
        # example: 20240529_1.jpg
        stem = filename.split(".")[0]
        date_str, camera_id = stem.split("_")

        # ---- extract annotations ----
        for annotation in image_entry.get("annotations", []):
            for result in annotation.get("result", []):
                if result["type"] != "keypointlabels":
                    continue

                val = result["value"]
                width = result["original_width"]
                height = result["original_height"]

                # convert percent -> pixels
                x_px = (val["x"] / 100.0) * width
                y_px = (val["y"] / 100.0) * height

                experiments[date_str][camera_id].append({
                    "x_px": x_px,
                    "y_px": y_px,
                    "x_pct": val["x"],
                    "y_pct": val["y"],
                    "image_width": width,
                    "image_height": height,
                    "camera_id": camera_id,
                    "image_name": filename
                })

    # ---- convert to requested output format ----
    experiment_dict = {}

    for date, cameras in experiments.items():
        dfs = []
        for camera_id, points in cameras.items():
            if points:  # safety check
                dfs.append(pd.DataFrame(points))
        experiment_dict[date] = dfs

    return experiment_dict

def plot_points_by_camera(experiment_dict, alpha=0.4, cmap_name="viridis"):
    """
    Plot all annotated points across all experiments, separated by camera.
    Points are colored by experiment order using a chromatic scale.

    Parameters
    ----------
    experiment_dict : dict
        Output of build_experiment_dict():
        {date: [df_cam1, df_cam2, ...], ...}

    alpha : float, optional
        Point transparency (default = 0.4)

    cmap_name : str, optional
        Matplotlib colormap name (default = "viridis")
    """

    # ---- sort experiments chronologically ----
    experiment_dates = sorted(experiment_dict.keys())
    n_experiments = len(experiment_dates)

    cmap = get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_experiments))

    # ---- collect data by camera ----
    camera_data = {}

    for exp_idx, date in enumerate(experiment_dates):
        for df in experiment_dict[date]:
            camera_id = df["camera_id"].iloc[0]

            if camera_id not in camera_data:
                camera_data[camera_id] = []

            camera_data[camera_id].append(
                (df, colors[exp_idx], date)
            )

    # ---- plotting ----
    for camera_id, datasets in camera_data.items():
        plt.figure()
        for df, color, date in datasets:
            plt.scatter(
                df["x_px"],
                df["y_px"],
                alpha=alpha,
                color=color,
                label=date
            )

        plt.title(f"Camera {camera_id}: All Experiments")
        plt.xlabel("X pixel")
        plt.ylabel("Y pixel")
        plt.gca().invert_yaxis()  # image-style coordinates
        plt.legend(title="Experiment Date", fontsize="small")
        plt.tight_layout()
        plt.show()

def plot_points_by_camera_4panel(experiment_dict, alpha=0.4, cmap_name="viridis"):
    """
    Plot annotated points across all experiments, grouped by camera.
    Cameras are displayed as 4 panels (2x2) per figure.
    Points are colored by experiment order using a chromatic scale.

    Parameters
    ----------
    experiment_dict : dict
        Output of build_experiment_dict():
        {date: [df_cam1, df_cam2, ...], ...}

    alpha : float, optional
        Point transparency (default = 0.4)

    cmap_name : str, optional
        Matplotlib colormap name (default = "viridis")
    """

    # ---- sort experiments chronologically ----
    experiment_dates = sorted(experiment_dict.keys())
    n_experiments = len(experiment_dates)

    cmap = get_cmap(cmap_name)
    colors = cmap(np.linspace(0, 1, n_experiments))

    # ---- collect data by camera ----
    camera_data = {}

    for exp_idx, date in enumerate(experiment_dates):
        for df in experiment_dict[date]:
            camera_id = df["camera_id"].iloc[0]

            camera_data.setdefault(camera_id, []).append(
                (df, colors[exp_idx], date)
            )

    camera_ids = sorted(camera_data.keys())
    n_cameras = len(camera_ids)

    # ---- figure batching: 4 cameras per figure ----
    cameras_per_fig = 4
    n_figs = ceil(n_cameras / cameras_per_fig)

    for fig_idx in range(n_figs):
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()

        start = fig_idx * cameras_per_fig
        end = min(start + cameras_per_fig, n_cameras)

        for ax_idx, cam_idx in enumerate(range(start, end)):
            ax = axes[ax_idx]
            camera_id = camera_ids[cam_idx]

            for df, color, date in camera_data[camera_id]:
                ax.scatter(
                    df["x_px"],
                    df["y_px"],
                    alpha=alpha,
                    color=color
                )

            ax.set_title(f"Camera {camera_id}")
            ax.set_xlabel("X pixel")
            ax.set_ylabel("Y pixel")
            ax.invert_yaxis()

        # ---- hide unused panels ----
        for ax in axes[end - start:]:
            ax.axis("off")

        fig.suptitle("Annotated Points Across Experiments", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def match_gcps_single_experiment(
    points_df,
    gcp_df,
    max_distance_px=None
):
    """
    Match GCPs to detected image points for ONE camera and ONE experiment.

    Parameters
    ----------
    points_df : pandas.DataFrame
        Detected points for one camera in one experiment.
        Must contain columns: ['x_px', 'y_px']

    gcp_df : pandas.DataFrame
        GCP reference table for that camera.
        Columns must be:
        ['Target', 'XPixels', 'YPixels', 'x_target', 'y_target']

    max_distance_px : float, optional
        Maximum allowed pixel distance for a match.

    Returns
    -------
    pandas.DataFrame
        Copy of points_df with added columns:
        ['Target', 'x_target', 'y_target', 'match_distance_px']
    """

    points = points_df.copy()
    gcps = gcp_df.copy()

    # initialize output columns
    points["Target"] = None
    points["x_target"] = np.nan
    points["y_target"] = np.nan
    points["match_distance_px"] = np.nan

    # indices of points not yet matched
    available_points = set(points.index)

    for _, gcp in gcps.iterrows():
        if not available_points:
            break

        px = gcp["XPixels"]
        py = gcp["YPixels"]

        idxs = list(available_points)
        dx = points.loc[idxs, "x_px"] - px
        dy = points.loc[idxs, "y_px"] - py
        distances = np.sqrt(dx**2 + dy**2)

        nearest_idx = distances.idxmin()
        min_dist = distances.loc[nearest_idx]

        if max_distance_px is not None and min_dist > max_distance_px:
            continue

        points.loc[nearest_idx, "Target"] = gcp["Target"]
        points.loc[nearest_idx, "x_target"] = gcp["x_target"]
        points.loc[nearest_idx, "y_target"] = gcp["y_target"]
        points.loc[nearest_idx, "match_distance_px"] = min_dist

        available_points.remove(nearest_idx)

    return points

def match_gcps_single_experiment_all_cameras(
    dfs_for_experiment,
    gcp_csv_map,
    max_distance_px=None
):
    """
    Match GCPs for all cameras within a single experiment.

    Parameters
    ----------
    dfs_for_experiment : list[pandas.DataFrame]
        List of camera DataFrames for one experiment date.

    gcp_csv_map : dict
        {camera_id: path_to_gcp_csv}

    max_distance_px : float, optional
        Maximum allowed match distance.

    Returns
    -------
    list[pandas.DataFrame]
        Same structure as input, but with GCPs assigned.
    """

    matched_dfs = []

    for df in dfs_for_experiment:
        camera_id = df["camera_id"].iloc[0]

        if camera_id not in gcp_csv_map:
            matched_dfs.append(df.copy())
            continue

        gcp_df = pd.read_csv(gcp_csv_map[camera_id])

        matched_df = match_gcps_single_experiment(
            df,
            gcp_df,
            max_distance_px=max_distance_px
        )

        matched_dfs.append(matched_df)

    return matched_dfs

def match_gcps_per_experiment(
    experiment_dict,
    gcp_csv_map,
    max_distance_px=None
):
    """
    Perform per-experiment, per-camera GCP matching.

    Parameters
    ----------
    experiment_dict : dict
        Output of build_experiment_dict():
        {date: [df_cam1, df_cam2, ...]}

    gcp_csv_map : dict
        {camera_id: path_to_gcp_csv}

    max_distance_px : float, optional
        Maximum allowed pixel distance.

    Returns
    -------
    dict
        Same structure as experiment_dict, but with GCPs assigned.
    """

    matched = {}

    for date, dfs in experiment_dict.items():
        matched[date] = match_gcps_single_experiment_all_cameras(
            dfs,
            gcp_csv_map,
            max_distance_px=max_distance_px
        )

    return matched

def plot_gcp_matching_qa(
    matched_df,
    gcp_df,
    camera_id,
    experiment_date,
    figsize=(12, 6)
):
    """
    QA plot for GCP matching for one camera and one experiment.

    Parameters
    ----------
    matched_df : pandas.DataFrame
        Output of match_gcps_single_experiment()
        (points with assigned GCPs)

    gcp_df : pandas.DataFrame
        GCP reference CSV for this camera.

    camera_id : str or int
        Camera identifier.

    experiment_date : str
        Experiment date (YYYYMMDD).

    figsize : tuple
        Figure size.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # ---- all detected points ----
    ax.scatter(
        matched_df["x_px"],
        matched_df["y_px"],
        s=15,
        color="gray",
        alpha=0.3,
        label="Detected points"
    )

    # ---- approximate GCP locations ----
    ax.scatter(
        gcp_df["XPixels"],
        gcp_df["YPixels"],
        color="red",
        marker="x",
        s=80,
        label="GCP (approx)"
    )

    # label approximate GCPs
    for _, gcp in gcp_df.iterrows():
        ax.text(
            gcp["XPixels"] + 5,
            gcp["YPixels"] + 5,
            gcp["Target"],
            color="red",
            fontsize=9
        )

    # ---- matched points + connectors ----
    matched_only = matched_df.dropna(subset=["Target"])

    for _, row in matched_only.iterrows():
        ax.scatter(
            row["x_px"],
            row["y_px"],
            s=60,
            edgecolor="k",
            facecolor="C0",
            zorder=3
        )

        ax.text(
            row["x_px"] + 5,
            row["y_px"] + 5,
            row["Target"],
            fontsize=9,
            color="C0"
        )

        # connector line
        gcp_row = gcp_df[gcp_df["Target"] == row["Target"]].iloc[0]

        ax.plot(
            [gcp_row["XPixels"], row["x_px"]],
            [gcp_row["YPixels"], row["y_px"]],
            linestyle="--",
            color="black",
            linewidth=1
        )

    ax.set_title(
        f"QA: Camera {camera_id}, Experiment {experiment_date}",
        fontsize=12
    )
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.invert_yaxis()
    ax.legend()
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

def plot_experiment_gcp_qa(
    matched_experiment_dfs,
    gcp_csv_map,
    experiment_date
):
    """
    Generate QA plots for all cameras in a single experiment.

    Parameters
    ----------
    matched_experiment_dfs : list[pandas.DataFrame]
        Output of match_gcps_single_experiment_all_cameras()

    gcp_csv_map : dict
        {camera_id: path_to_gcp_csv}

    experiment_date : str
        Experiment date.
    """

    for df in matched_experiment_dfs:
        camera_id = df["camera_id"].iloc[0]

        if camera_id not in gcp_csv_map:
            continue

        gcp_df = pd.read_csv(gcp_csv_map[camera_id])

        plot_gcp_matching_qa(
            matched_df=df,
            gcp_df=gcp_df,
            camera_id=camera_id,
            experiment_date=experiment_date
        )

gcps = build_experiment_dict("C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/GCP selection/annotated_gcps.json")

print(gcps)
plot_points_by_camera_4panel(gcps)

gcps_csv_map = {"1": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/GCP selection/original GCPs/Cam1_gcps.csv",
                "2": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/GCP selection/original GCPs/Cam2_gcps.csv",
                "3": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/GCP selection/original GCPs/Cam3_gcps.csv",
                "4": "C:/Users/josie/OneDrive - UCB-O365/Wood Tracking/GCP selection/original GCPs/Cam4_gcps.csv"}

gcp_assigned = match_gcps_per_experiment(gcps, gcps_csv_map, 30)

print(gcp_assigned)


for date, dfs in gcp_assigned.items():
    plot_experiment_gcp_qa(
        gcp_assigned[date],
        gcps_csv_map,
        experiment_date=date
    )