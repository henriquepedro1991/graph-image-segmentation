import os
import json
import glob
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.segmentation import slic
from skimage.util import img_as_float
import igraph as ig
from hedonic.hedonic import HedonicGame


def plot_metrics_and_memory():
    """
    Reads the combined metadata from 'all_metadata.json' and creates:
      - A bar graph comparing metrics by method.
      - A bar graph showing average memory usage per method.
    """
    combined_json_path = "results/metadata/all_metadata.json"
    with open(combined_json_path, "r") as file:
        combined_data = json.load(file)

    metrics_data = []
    memory_data = []

    # Extract metrics and memory usage from each JSON entry
    for key, value in combined_data.items():
        file_name = value["file_name"]
        for method, metrics in value["metrics"].items():
            metrics_data.append(
                {
                    "file_name": file_name,
                    "method": method,
                    "ami": metrics.get("ami", None),
                    "rand_index": metrics.get("rand_index", None),
                    "f1_score": metrics.get("f1_score", None),
                    "iou": metrics.get("iou", None),
                }
            )
        mem_usage = value.get("memory_usage", {})
        for method, mem in mem_usage.items():
            memory_data.append(
                {"file_name": file_name, "method": method, "memory_usage": mem}
            )

    df_metrics = pd.DataFrame(metrics_data)
    numeric_cols = ["ami", "rand_index", "f1_score", "iou"]
    for col in numeric_cols:
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")

    grouped = df_metrics.groupby("method", as_index=False)[numeric_cols].mean()
    melted = grouped.melt(
        id_vars="method", value_vars=numeric_cols, var_name="Metric", value_name="Value"
    )

    output_folder = "results/bar_graphs"
    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="Metric", y="Value", hue="method", errorbar="sd")
    plt.title("Comparison of Metrics by Method")
    plt.xlabel("Metric")
    plt.ylabel("Value (mean ± SD)")
    plt.xticks(rotation=45)
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plot_filename = os.path.join(output_folder, "all_metrics_bar_graph.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Unified metrics bar graph saved in: {plot_filename}")

    df_memory = pd.DataFrame(memory_data)
    df_memory["memory_usage"] = pd.to_numeric(
        df_memory["memory_usage"], errors="coerce"
    )
    grouped_mem = df_memory.groupby("method", as_index=False)["memory_usage"].agg(
        {"avg_memory": "mean", "std_memory": "std"}
    )

    plt.figure(figsize=(8, 6))
    x = np.arange(len(grouped_mem["method"]))
    plt.bar(
        x,
        grouped_mem["avg_memory"],
        yerr=grouped_mem["std_memory"],
        capsize=5,
        color=["blue", "orange", "green", "red", "purple"],
    )
    plt.xticks(x, grouped_mem["method"])
    plt.xlabel("Method")
    plt.ylabel("Memory Usage (MB)")
    plt.title("Average Memory Usage by Method")
    plt.tight_layout()
    mem_plot_filename = os.path.join(output_folder, "memory_usage_by_method.png")
    plt.savefig(mem_plot_filename, dpi=300)
    plt.close()
    print(f"Memory usage bar graph saved in: {mem_plot_filename}")


def plot_processing_times():
    """
    Reads processing times from 'all_metadata.json' and creates a bar graph showing
    average processing time (with standard deviation) for each segmentation method.
    """
    with open("results/metadata/all_metadata.json", "r") as f:
        metadata = json.load(f)

    methods = ["leiden", "hedonic", "louvain", "label_propagation", "infomap"]
    processing_times = {m: [] for m in methods}
    for key in metadata:
        for m in methods:
            processing_times[m].append(metadata[key]["processing_time"][m])

    for m in methods:
        processing_times[m] = np.array(processing_times[m])

    avg_times = [np.mean(processing_times[m]) for m in methods]
    std_times = [np.std(processing_times[m]) for m in methods]

    x = np.arange(len(methods))
    width = 0.5
    plt.figure(figsize=(8, 6))
    plt.bar(
        x,
        avg_times,
        yerr=std_times,
        width=width,
        color=["blue", "orange", "green", "red", "purple"],
        capsize=5,
    )
    plt.xticks(x, methods)
    plt.ylabel("Processing Time (seconds)")
    plt.title("Average Processing Time per Method")
    plt.tight_layout()

    output_folder = "results/performance"
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, "processing_time_comparison.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Processing time plot saved at: {plot_path}")


# The following functions (with a 'graph_' prefix) are used for community segmentation plots.
def graph_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"File not found or invalid: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def graph_create_superpixels(image, n_segments=500, compactness=30):
    image_float = img_as_float(image)
    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=1,
        channel_axis=-1,
    )
    return segments


def graph_create_graph_from_superpixels(image, segments):
    height, width = image.shape[:2]
    n_superpixels = np.max(segments)
    graph = ig.Graph()
    graph.add_vertices(n_superpixels)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sp_weights = np.zeros(n_superpixels, dtype=float)
    sp_counts = np.zeros(n_superpixels, dtype=int)
    for i in range(height):
        for j in range(width):
            sp_id = segments[i, j] - 1
            sp_weights[sp_id] += gray[i, j] / 255.0
            sp_counts[sp_id] += 1
    sp_weights /= sp_counts
    graph.vs["weight"] = sp_weights

    edges = set()
    edge_weights = []
    for i in range(height):
        for j in range(width):
            sp_id = segments[i, j] - 1
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                ni, nj = i + dx, j + dy
                if 0 <= ni < height and 0 <= nj < width:
                    neigh_sp_id = segments[ni, nj] - 1
                    if sp_id != neigh_sp_id:
                        edge = tuple(sorted((sp_id, neigh_sp_id)))
                        if edge not in edges:
                            edges.add(edge)
                            diff = abs(sp_weights[sp_id] - sp_weights[neigh_sp_id])
                            edge_weights.append(diff)
    graph.add_edges(list(edges))
    graph.es["weight"] = edge_weights
    return graph


def graph_get_connected_components_partition(graph):
    comps = graph.components()
    membership = [None] * graph.vcount()
    for idx, comp in enumerate(comps):
        for v in comp:
            membership[v] = idx
    return membership


def graph_segment_graph_global_optimum(graph, resolution):
    n = graph.vcount()
    comps = graph.components()
    if abs(resolution) < 1e-9:
        membership = graph_get_connected_components_partition(graph)
        return membership
    elif abs(resolution - 1.0) < 1e-9:
        return list(range(n))
    else:
        return None


def graph_random_initial_membership(n_vertices):
    return np.random.randint(0, n_vertices, size=n_vertices).tolist()


def graph_run_hedonic_multiple_times(graph, resolution, runs=10):
    n = graph.vcount()
    expected = 1 + resolution * (n - 1)
    best_partition = None
    best_diff = float("inf")
    for _ in range(runs):
        init_membership = graph_random_initial_membership(n)
        hedonic_game = HedonicGame(graph)
        partition = hedonic_game.community_hedonic(
            resolution=resolution, initial_membership=init_membership
        )
        num_com = len(partition)
        diff = abs(num_com - expected)
        if diff < best_diff:
            best_diff = diff
            best_partition = partition
    return best_partition


def graph_compute_partition(graph, resolution, runs=10, use_queue=False):
    membership_global = graph_segment_graph_global_optimum(graph, resolution)
    if membership_global is not None:
        comm_dict = defaultdict(list)
        for v_idx, comm_id in enumerate(membership_global):
            comm_dict[comm_id].append(v_idx)
        partition = list(comm_dict.values())
    else:
        if use_queue:
            hedonic_game = HedonicGame(graph)
            partition = hedonic_game.community_hedonic_queue(
                resolution=resolution, initial_membership=None
            ).membership
        else:
            partition = graph_run_hedonic_multiple_times(graph, resolution, runs=runs)
    return partition


def plot_communities_vs_alpha():
    """
    For each image in 'images/original', compute the community partition over a range of
    alpha (resolution) values and plot:
      - One curve per image.
      - A mean ± standard deviation curve.
    """
    input_dir = "images/original"
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print("No images found in the directory:", input_dir)
        return

    output_folder = "results/analysis_log_scale"
    os.makedirs(output_folder, exist_ok=True)

    alpha_values = np.logspace(-2, 0, 30)
    communities_data = np.zeros((len(image_paths), len(alpha_values)))

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")
        original_image = graph_preprocess_image(img_path)
        segments = graph_create_superpixels(
            original_image, n_segments=280, compactness=2
        )
        graph_obj = graph_create_graph_from_superpixels(original_image, segments)
        for j, alpha in enumerate(alpha_values):
            partition = graph_compute_partition(graph_obj, alpha, runs=10)
            communities_data[i, j] = len(partition)
            print(f"   Alpha {alpha:.3f} -> {len(partition)} communities")

    plt.figure(figsize=(8, 6))
    for i in range(len(image_paths)):
        plt.plot(
            alpha_values,
            communities_data[i, :],
            "-o",
            markersize=3,
            linewidth=1,
            label=f"Image {i+1}",
        )
    plt.xscale("log")
    plt.xlabel("Alpha (resolution) [log scale]")
    plt.ylabel("Number of Communities")
    plt.title("Communities vs. Alpha (Log Scale) - Individual Images")
    plt.grid(True)
    plt.tight_layout()
    plot_path_lines = os.path.join(
        output_folder, "communities_vs_alpha_all_lines_2.png"
    )
    plt.savefig(plot_path_lines)
    plt.show()
    print(f"Plot (individual lines) saved at: {plot_path_lines}")

    mean_communities = communities_data.mean(axis=0)
    std_communities = communities_data.std(axis=0)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        alpha_values,
        mean_communities,
        yerr=std_communities,
        fmt="o-",
        ecolor="red",
        capsize=5,
        markersize=4,
        linewidth=1,
    )
    plt.xscale("log")
    plt.xlabel("Alpha (resolution) [log scale]")
    plt.ylabel("Number of Communities (mean ± std)")
    plt.title("Communities vs. Alpha (Log Scale) with Confidence Intervals")
    plt.grid(True)
    plt.tight_layout()
    plot_path_mean_std = os.path.join(
        output_folder, "communities_vs_alpha_mean_std_2.png"
    )
    plt.savefig(plot_path_mean_std)
    plt.show()
    print(f"Plot (mean ± std) saved at: {plot_path_mean_std}")


def plot_communities_vs_density_multiplier():
    """
    For each image in 'images/original', compute the community partition over a range of
    density-based multipliers (alpha = density * multiplier) and plot:
      - One curve per image.
      - A mean ± standard deviation curve.
    """
    input_dir = "images/original"
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_paths:
        print("No images found in the directory:", input_dir)
        return

    output_folder = "results/density_multipliers"
    os.makedirs(output_folder, exist_ok=True)

    multipliers = np.logspace(-2, 2, 30)
    communities_data = np.zeros((len(image_paths), len(multipliers)))

    for i, img_path in enumerate(image_paths):
        print(f"Processing {img_path}...")
        original_image = graph_preprocess_image(img_path)
        segments = graph_create_superpixels(
            original_image, n_segments=280, compactness=2
        )
        graph_obj = graph_create_graph_from_superpixels(original_image, segments)
        density = graph_obj.density()
        for j, m in enumerate(multipliers):
            alpha = density * m
            partition = graph_compute_partition(graph_obj, alpha, runs=10)
            communities_data[i, j] = len(partition)

    plt.figure(figsize=(8, 6))
    for i in range(len(image_paths)):
        plt.plot(multipliers, communities_data[i, :], "-o", markersize=3, linewidth=1)
    plt.xscale("log")
    plt.xlabel("Multiplier (alpha = density * multiplier) [log scale]")
    plt.ylabel("Number of Communities")
    plt.title("Communities vs. Density-based Multiplier (Each Image)")
    plt.grid(True)
    plt.tight_layout()
    plot_path_lines = os.path.join(
        output_folder, "communities_vs_density_multiplier_all_lines_2.png"
    )
    plt.savefig(plot_path_lines)
    plt.show()
    print(f"Plot (all lines) saved at: {plot_path_lines}")

    mean_communities = communities_data.mean(axis=0)
    std_communities = communities_data.std(axis=0)
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        multipliers,
        mean_communities,
        yerr=std_communities,
        fmt="o-",
        ecolor="red",
        capsize=5,
        markersize=4,
        linewidth=1,
    )
    plt.xscale("log")
    plt.xlabel("Multiplier (alpha = density * multiplier) [log scale]")
    plt.ylabel("Number of Communities (mean ± std)")
    plt.title("Communities vs. Density-based Multiplier (Log Scale)")
    plt.grid(True)
    plt.tight_layout()
    plot_path_mean_std = os.path.join(
        output_folder, "communities_vs_density_multiplier_mean_std_2.png"
    )
    plt.savefig(plot_path_mean_std)
    plt.show()
    print(f"Plot (mean ± std) saved at: {plot_path_mean_std}")


def run_all_methods_graphs():
    print("Plotting metrics and memory usage graphs...")
    plot_metrics_and_memory()
    print("Plotting processing time graph...")
    plot_processing_times()
    print("Plotting communities vs. alpha graphs...")
    plot_communities_vs_alpha()
    print("Plotting communities vs. density multiplier graphs...")
    plot_communities_vs_density_multiplier()
