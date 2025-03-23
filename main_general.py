import os
import time
import json
from scipy.io import loadmat
from functions import (
    generate_segmented_images,
    all_methods_graphs,
)

###########################################
# Main Processing For All Methods
###########################################
if __name__ == "__main__":
    input_folder = "images/original/"
    groundtruth_folder = "images/groundtruth/"
    output_folder = "results/"
    n_segments = 280
    compactness = 2
    methods = ["leiden", "hedonic", "louvain", "label_propagation", "infomap"]
    resolution = 10

    os.makedirs(os.path.join(output_folder, "graphs"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "segmented_images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "metadata"), exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            groundtruth_path = os.path.join(groundtruth_folder, f"{base_name}.mat")

            print(f"Processing image: {filename}")
            start_time = time.time()

            original_image = generate_segmented_images.preprocess_image(image_path)
            segments = generate_segmented_images.create_superpixels(
                original_image, n_segments, compactness
            )

            boundary_output_path = os.path.join(
                output_folder,
                "segmented_images",
                f"{base_name}_superpixel_boundaries.png",
            )
            generate_segmented_images.save_superpixel_boundaries(
                original_image, segments, boundary_output_path
            )

            graph = generate_segmented_images.create_graph_from_superpixels(
                original_image, segments
            )

            graph_output_path = os.path.join(
                output_folder, "graphs", f"{base_name}_graph.png"
            )
            generate_segmented_images.save_graph_as_image(graph, graph_output_path)

            if os.path.exists(groundtruth_path):
                groundtruth_mat = loadmat(groundtruth_path)
                groundtruth = groundtruth_mat["groundTruth"][0][0][0][0][
                    "Segmentation"
                ].astype(int)
            else:
                groundtruth = None
                print(f"Groundtruth not found for {filename}. Metrics will be skipped.")

            metadata = {
                "file_name": filename,
                "processing_time": {},
                "memory_usage": {},
                "metrics": {},
            }

            for method in methods:
                print(f"Segmenting using the {method} method...")
                method_start_time = time.time()

                (label_image, visual_image, partition), method_proc_time, method_mem = (
                    generate_segmented_images.measure_performance(
                        generate_segmented_images.segment_image_with_graph,
                        original_image,
                        segments,
                        graph,
                        method,
                        resolution,
                    )
                )

                segmented_output_path = os.path.join(
                    output_folder,
                    "segmented_images",
                    f"{base_name}_segmented_{method}.png",
                )
                generate_segmented_images.save_segmented_image(
                    visual_image, segmented_output_path
                )

                segmented_graph_output_path = os.path.join(
                    output_folder, "graphs", f"{base_name}_graph_{method}.png"
                )
                generate_segmented_images.save_graph_colored_by_partition(
                    graph, partition, segmented_graph_output_path
                )

                if groundtruth is not None:
                    metrics = generate_segmented_images.compute_metrics(
                        label_image, groundtruth
                    )
                    metadata["metrics"][method] = metrics

                metadata["processing_time"][method] = method_proc_time
                metadata["memory_usage"][method] = method_mem

            metadata_output_path = os.path.join(
                output_folder, "metadata", f"{base_name}.json"
            )
            with open(metadata_output_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)

            print(f"Metadata saved to {metadata_output_path}")
            print(
                f"Image {filename} processed in {time.time() - start_time:.2f} seconds."
            )

    print("Graph-based segmentation completed for all images.")

    # After processing images, generate all graphs using functions from all_methods_graphs
    generate_segmented_images.combine_json_files(
        "results/metadata", os.path.join("results", "all_metadata.json")
    )

    all_methods_graphs.plot_metrics_and_memory()
    all_methods_graphs.plot_processing_times()
    all_methods_graphs.plot_communities_vs_alpha()
    all_methods_graphs.plot_communities_vs_density_multiplier()
