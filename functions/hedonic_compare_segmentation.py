import time
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


# ----------------------------
# Performance Measurement Function
# ----------------------------
def measure_performance(func, *args, repeat=10, **kwargs):
    process = psutil.Process()
    mem_before = process.memory_info().rss
    start_time = time.time()
    for _ in range(repeat):
        result = func(*args, **kwargs)
    mem_after = process.memory_info().rss
    end_time = time.time()
    processing_time = end_time - start_time
    mem_peak = max(mem_before, mem_after) / (1024 * 1024)  # in MB
    return result, processing_time, mem_peak


# ----------------------------
# Preprocessing and Graph Functions
# ----------------------------
def preprocess_image(image_path):
    """Load and convert the image to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def create_superpixels(image, n_segments=280, compactness=2):
    """Generate superpixels using SLIC."""
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
    """Overlay superpixel boundaries on the image and save the result."""
    boundary_image = mark_boundaries(img_as_float(image), segments, color=(1, 1, 1))
    plt.figure(figsize=(12, 8))
    plt.imshow(boundary_image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Superpixel boundaries image saved to {output_path}")


def create_graph_from_superpixels(image, segments):
    """Build an igraph graph from superpixels based on 8-neighbor adjacency."""
    height, width = image.shape[:2]
    n_superpixels = np.max(segments)
    graph = ig.Graph()
    graph.add_vertices(n_superpixels)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    weights = np.zeros(n_superpixels, dtype=np.float32)
    counts = np.zeros(n_superpixels, dtype=np.int32)
    for i in range(height):
        for j in range(width):
            sp_id = segments[i, j] - 1
            weights[sp_id] += gray[i, j] / 255.0
            counts[sp_id] += 1
    weights /= counts
    graph.vs["weight"] = weights

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
                    neighbor_sp_id = segments[ni, nj] - 1
                    if sp_id != neighbor_sp_id:
                        edge = tuple(sorted((sp_id, neighbor_sp_id)))
                        if edge not in edges:
                            edges.add(edge)
                            weight_diff = abs(weights[sp_id] - weights[neighbor_sp_id])
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
    """Save a graph visualization with vertices colored according to the partition."""
    membership = partition.membership
    unique_communities = sorted(set(membership))
    num_communities = len(unique_communities)
    community_to_index = {comm: i for i, comm in enumerate(unique_communities)}
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


def segment_image_with_graph(image, segments, graph, method="leiden", resolution=1.0):
    """
    Segment the image using the specified community detection method.
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
        partition = hedonic_game.community_hedonic_queue(resolution=resolution)
    else:
        raise ValueError("Unsupported segmentation method.")
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
    """Save the segmented image visualization."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Segmented image saved to {output_path}")


def compute_metrics(segmented_labels, groundtruth_labels):
    """
    Compute segmentation metrics: AMI, Rand Index, F1 Score, and IoU.
    If shapes differ, ground truth is resized to match.
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
    rand_idx = rand_score(gt_flat, seg_flat)
    groundtruth_bin = (groundtruth_labels > 0).astype(int).ravel()
    segmented_bin = (segmented_labels > 0).astype(int).ravel()
    intersection = np.logical_and(groundtruth_bin, segmented_bin).sum()
    union = np.logical_or(groundtruth_bin, segmented_bin).sum()
    iou = intersection / union if union != 0 else 0.0
    f1 = f1_score(groundtruth_bin, segmented_bin, average="macro")
    return {"ami": ami, "rand_index": rand_idx, "f1_score": f1, "iou": iou}
