import os
import time
import json
import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, rand_score, f1_score
from hedonic.hedonic import HedonicGame
import igraph as ig
import random
import psutil


###########################################
# Image Preprocessing and Graph Creation Functions
###########################################
def preprocess_image(image_path):
    """Load the image and convert from BGR to RGB."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def create_superpixels(image, n_segments=500, compactness=30):
    """Generate superpixels from the image using SLIC."""
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


def save_superpixel_boundaries(image, segments, output_path):
    """Save the image with superpixel boundaries overlaid."""
    boundary_image = mark_boundaries(img_as_float(image), segments, color=(1, 1, 1))
    plt.figure(figsize=(12, 8))
    plt.imshow(boundary_image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Superpixel boundaries image saved to {output_path}")


def create_graph_from_superpixels(image, segments):
    """
    Create a graph where each vertex corresponds to a superpixel.
    Edges are determined based on 8-neighbor adjacency.
    """
    height, width = image.shape[:2]
    n_superpixels = np.max(segments)

    graph = ig.Graph()
    graph.add_vertices(n_superpixels)

    weights = np.zeros(n_superpixels, dtype=np.float32)
    counts = np.zeros(n_superpixels, dtype=np.int32)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    for i in range(height):
        for j in range(width):
            segment_id = segments[i, j] - 1
            weights[segment_id] += gray_image[i, j] / 255.0
            counts[segment_id] += 1

    weights /= counts
    graph.vs["weight"] = weights

    edges = set()
    edge_weights = []

    for i in range(height):
        for j in range(width):
            segment_id = segments[i, j] - 1
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
                    neighbor_segment_id = segments[ni, nj] - 1
                    if segment_id != neighbor_segment_id:
                        edge = tuple(sorted((segment_id, neighbor_segment_id)))
                        if edge not in edges:
                            edges.add(edge)
                            weight_diff = abs(
                                weights[segment_id] - weights[neighbor_segment_id]
                            )
                            edge_weights.append(weight_diff)

    graph.add_edges(list(edges))
    graph.es["weight"] = edge_weights
    return graph


def save_graph_as_image(graph, file_path):
    """Save a basic visualization of the graph."""
    layout = graph.layout("fr")
    ig.plot(
        graph,
        target=file_path,
        layout=layout,
        vertex_size=20,
        vertex_label=None,
        edge_width=0.5,
        edge_color="gray",
        vertex_color="red",
    )
    print(f"Graph image saved to {file_path}")


def save_graph_colored_by_partition(graph, partition, file_path):
    """
    Save a graph visualization with vertices colored by community membership.
    Reindexes membership IDs to avoid indexing errors.
    """
    membership = partition.membership
    unique_communities = sorted(set(membership))
    num_communities = len(unique_communities)
    community_to_index = {
        community_id: i for i, community_id in enumerate(unique_communities)
    }
    random.seed(0)
    color_list = [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(num_communities)
    ]
    vertex_colors = [color_list[community_to_index[m]] for m in membership]
    layout = graph.layout("fr")
    ig.plot(
        graph,
        target=file_path,
        layout=layout,
        vertex_size=20,
        vertex_label=None,
        edge_width=0.5,
        edge_color="gray",
        vertex_color=vertex_colors,
    )
    print(f"Colored graph image saved to {file_path}")


###########################################
# Segmentation and Metric Functions
###########################################
def segment_image_with_graph(image, segments, graph, method="leiden", resolution=1.0):
    """
    Perform segmentation using a community detection method.
    Returns (label_image, visual_image, partition).
    """
    if method == "leiden":
        import leidenalg

        partition = leidenalg.find_partition(
            graph, leidenalg.CPMVertexPartition, resolution_parameter=resolution
        )
    elif method == "louvain":
        partition = graph.community_multilevel()
    elif method == "label_propagation":
        partition = graph.community_label_propagation()
    elif method == "infomap":
        partition = graph.community_infomap()
    elif method == "hedonic":
        hedonic_game = HedonicGame(graph)
        partition = hedonic_game.community_hedonic()
    else:
        raise ValueError(
            "Unsupported method. Use 'leiden', 'louvain', 'label_propagation', 'infomap', or 'hedonic'."
        )

    height, width = segments.shape
    label_image = np.zeros((height, width), dtype=np.int32)
    for community_index, vertices in enumerate(partition):
        for v in vertices:
            label_image[segments == (v + 1)] = community_index

    n_communities = len(partition)
    colors = np.random.rand(n_communities, 3)
    visual_image = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            cid = label_image[i, j]
            visual_image[i, j] = colors[cid]
    visual_image = (visual_image * 255).astype(np.uint8)
    return label_image, visual_image, partition


def save_segmented_image(image, output_path):
    """Save the segmented RGB visualization image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Segmented image saved to {output_path}")


def compute_metrics(segmented_labels, groundtruth_labels):
    """
    Compute segmentation metrics.
    """
    if segmented_labels.shape != groundtruth_labels.shape:
        groundtruth_labels = resize(
            groundtruth_labels,
            segmented_labels.shape,
            mode="edge",
            preserve_range=True,
            anti_aliasing=False,
        ).astype(int)
    gt_flat = groundtruth_labels.ravel()
    seg_flat = segmented_labels.ravel()
    ami = adjusted_mutual_info_score(gt_flat, seg_flat)
    rand_index = rand_score(gt_flat, seg_flat)
    groundtruth_bin = (groundtruth_labels > 0).astype(int).ravel()
    segmented_bin = (segmented_labels > 0).astype(int).ravel()
    intersection = np.logical_and(groundtruth_bin, segmented_bin).sum()
    union = np.logical_or(groundtruth_bin, segmented_bin).sum()
    iou = intersection / union if union != 0 else 0.0
    f1 = f1_score(groundtruth_bin, segmented_bin, average="macro")
    return {
        "ami": ami,
        "rand_index": rand_index,
        "f1_score": f1,
        "iou": iou,
    }


def combine_json_files(folder_path, output_file):
    """
    Combines all JSON files in a given folder and saves the combined data into a single JSON file.
    """
    combined_data = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            file_key = file_name.replace(".json", "")
            combined_data[file_key] = data

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(combined_data, file, indent=4)

    print(f"All JSON files have been combined and saved in '{output_file}'.")


###########################################
# Function to measure performance (time and memory usage)
###########################################
def measure_performance(func, *args, repeat=10, **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()

    for _ in range(repeat):
        result = func(*args, **kwargs)

    mem_after = process.memory_info().rss
    end_time = time.time()

    processing_time = end_time - start_time
    mem_peak = max(mem_before, mem_after) / (1024 * 1024)  # Convert bytes to MB

    return result, processing_time, mem_peak
