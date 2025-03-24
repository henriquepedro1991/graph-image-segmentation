import cv2
import igraph as ig
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def preprocess_image(image_path):
    """Pre-process the image to enhance contrast for better segmentation."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def create_superpixels(image, n_segments=500, compactness=30):
    """Generate superpixels from the image."""
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


def create_graph_from_superpixels(image, segments):
    """Create a graph from superpixels where vertices are superpixels and edges are based on adjacency."""
    height, width = image.shape[:2]
    n_superpixels = np.max(segments)

    # Create graph
    graph = ig.Graph()
    graph.add_vertices(n_superpixels)

    # Compute vertex weights (average intensity for each superpixel)
    weights = np.zeros(n_superpixels, dtype=np.float32)
    counts = np.zeros(n_superpixels, dtype=np.int32)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    for i in range(height):
        for j in range(width):
            segment_id = segments[i, j] - 1
            weights[segment_id] += gray_image[i, j] / 255.0  # Normalize to [0, 1]
            counts[segment_id] += 1

    weights /= counts
    graph.vs["weight"] = weights

    # Create edges based on adjacency
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
    """Save the graph as an image."""
    layout = graph.layout("fr")  # Fruchterman-Reingold layout for visualization
    ig.plot(
        graph,
        target=file_path,
        layout=layout,
        vertex_size=20,
        vertex_label=None,
        edge_width=0.5,
        edge_color="gray",
    )
    print(f"Graph image saved to {file_path}")


def segment_image_with_graph(image, segments, graph, method="leiden", resolution=1.0):
    """Segment the image using graph-based community detection."""
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
        from hedonic.hedonic import HedonicGame

        hedonic_game = HedonicGame(graph)
        partition = hedonic_game.community_hedonic()
    else:
        raise ValueError(
            "Unsupported method. Use 'leiden', 'louvain', 'label_propagation', 'infomap', or 'hedonic'."
        )

    # Create a color map for the communities
    n_communities = len(partition)
    colors = np.random.rand(n_communities, 3)
    community_map = {
        vertex: colors[community]
        for community, vertices in enumerate(partition)
        for vertex in vertices
    }

    # Map colors back to the original segments
    segmented_image = np.zeros_like(image, dtype=np.float32)
    height, width = segments.shape
    for i in range(height):
        for j in range(width):
            segment_id = segments[i, j] - 1
            segmented_image[i, j] = community_map[segment_id]

    return (segmented_image * 255).astype(np.uint8)


def save_segmented_image(image, output_path):
    """Save the segmented image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Segmented image saved to {output_path}")


def refine_clusters(segmented_image, segments, min_cluster_size=200):
    """Combine small clusters into neighboring larger clusters."""
    from scipy.ndimage import label

    labeled, num_features = label(segmented_image[:, :, 0] > 0)
    sizes = np.bincount(labeled.ravel())
    mask = sizes < min_cluster_size
    for i in range(1, num_features + 1):
        if mask[i]:
            segmented_image[labeled == i] = [0, 0, 0]
    return segmented_image
