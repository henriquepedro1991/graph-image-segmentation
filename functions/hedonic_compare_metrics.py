import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.segmentation import slic
from skimage.util import img_as_float
import igraph as ig
from hedonic.hedonic import HedonicGame

# Importa as métricas do sklearn
from sklearn.metrics import (
    adjusted_mutual_info_score,
    rand_score,
    f1_score,
    jaccard_score,
)


# ----------------------------
# Preprocessing and Graph Creation Functions
# ----------------------------
def compute_edge_density(graph):
    """Calculate the graph density: number of edges divided by number of vertices."""
    n = graph.vcount()
    density = len(graph.es) / n
    return density


# ----------------------------
# Segmentation Functions
# ----------------------------
def get_connected_components_partition(graph):
    """
    Return the partition where each connected component is a community.
    """
    comps = graph.components()
    membership = [None] * graph.vcount()
    for idx, comp in enumerate(comps):
        for v in comp:
            membership[v] = idx
    return membership


def segment_graph_global_optimum(graph, resolution):
    """
    For extreme values:
      - If resolution == 0: if the graph is connected, return a single community;
        otherwise, each disconnected component is a community.
      - If resolution == 1: return a partition where each node is its own community.
    Otherwise, return None.
    """
    n = graph.vcount()
    comps = graph.components()
    if abs(resolution) < 1e-9:
        membership = get_connected_components_partition(graph)
        return membership
    elif abs(resolution - 1.0) < 1e-9:
        membership = list(range(n))
        return membership
    else:
        return None


def random_initial_membership(n_vertices):
    """Generate a random initial partition."""
    return np.random.randint(0, n_vertices, size=n_vertices).tolist()


def run_hedonic_multiple_times(graph, resolution, runs=10):
    """
    Run 'community_hedonic' multiple times with random initial partitions and return
    the partition whose number of communities is closest to the theoretical value:
         expected = 1 + resolution*(n-1)
    """
    n = graph.vcount()
    expected = 1 + resolution * (n - 1)
    best_partition = None
    best_diff = float("inf")
    for _ in range(runs):
        init_membership = random_initial_membership(n)
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


def label_to_image(partition, segments):
    """
    Convert the partition (list of lists with vertex indices) into a 2D label image.
    """
    height, width = segments.shape
    label_image = np.zeros((height, width), dtype=int)
    for cid, vertices in enumerate(partition):
        for v in vertices:
            label_image[segments == (v + 1)] = cid
    return label_image


def color_image_from_labels(label_image, base_image):
    """
    Create a color (RGB) image from the label map.
    Clusters are ordered canonically based on the mean horizontal coordinate of the pixels in each cluster,
    ensuring consistent colors between frames.
    """
    h, w = label_image.shape
    unique_labels = np.unique(label_image)
    centroids = {}
    for label in unique_labels:
        coords = np.column_stack(np.where(label_image == label))
        centroids[label] = coords[:, 1].mean()  # Mean of column indices
    sorted_labels = sorted(unique_labels, key=lambda l: centroids[l])
    np.random.seed(42)  # fix seed for consistent colors
    colors = np.random.rand(len(sorted_labels), 3)
    mapping = {label: idx for idx, label in enumerate(sorted_labels)}

    visual = np.zeros_like(base_image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            orig_label = label_image[i, j]
            new_idx = mapping[orig_label]
            visual[i, j] = colors[new_idx]
    visual = (visual * 255).astype(np.uint8)
    return visual


def segment_image(graph, segments, original_image, resolution, runs=10):
    """
    Segment the image based on the given resolution.
    If resolution is 0 or 1, use the global partition; otherwise, use local search.
    """
    membership_global = segment_graph_global_optimum(graph, resolution)
    if membership_global is not None:
        from collections import defaultdict

        comm_dict = defaultdict(list)
        for v_idx, comm_id in enumerate(membership_global):
            comm_dict[comm_id].append(v_idx)
        partition = list(comm_dict.values())
    else:
        partition = run_hedonic_multiple_times(graph, resolution, runs=runs)
    label_img = label_to_image(partition, segments)
    visual_img = color_image_from_labels(label_img, original_image)
    return label_img, visual_img


# ----------------------------
# Function to Overlay Resolution Value on Image
# ----------------------------
def add_resolution_text(image, resolution):
    """
    Overlay the resolution value on the image (RGB) and return the modified image.
    """
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    text = f"Res = {resolution:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 10, text_size[1] + 10
    cv2.putText(img_bgr, text, (x, y), font, font_scale, (0, 255, 0), thickness)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
