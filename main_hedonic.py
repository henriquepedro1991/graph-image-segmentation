import os
import time
import json
import cv2
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import imageio
from functions import (
    hedonic_alpha_images,
    hedonic_compare_metrics,
    hedonic_compare_segmentation,
)
from sklearn.metrics import (
    adjusted_mutual_info_score,
    rand_score,
    f1_score,
    jaccard_score,
)


###########################################
# Main Processing For Hedonic Method
###########################################
def main():
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
        if filename.lower().endswith((".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            groundtruth_path = os.path.join(groundtruth_folder, f"{base_name}.mat")

            print(f"Processing image: {filename}")
            start_time = time.time()
            original_image = hedonic_compare_segmentation.preprocess_image(image_path)
            segments = hedonic_compare_segmentation.create_superpixels(
                original_image, n_segments, compactness
            )
            boundary_output_path = os.path.join(
                output_folder,
                "segmented_images",
                f"{base_name}_superpixel_boundaries.png",
            )
            graph = hedonic_compare_segmentation.create_graph_from_superpixels(
                original_image, segments
            )
            graph_output_path = os.path.join(
                output_folder, "graphs", f"{base_name}_graph.png"
            )
            hedonic_compare_segmentation.save_graph_as_image(graph, graph_output_path)
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
                    hedonic_compare_segmentation.measure_performance(
                        hedonic_compare_segmentation.segment_image_with_graph,
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
                hedonic_compare_segmentation.save_segmented_image(
                    visual_image, segmented_output_path
                )
                segmented_graph_output_path = os.path.join(
                    output_folder, "graphs", f"{base_name}_graph_{method}.png"
                )
                hedonic_compare_segmentation.save_graph_colored_by_partition(
                    graph, partition, segmented_graph_output_path
                )
                if groundtruth is not None:
                    metrics = hedonic_compare_segmentation.compute_metrics(
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

    # -----------------------------------------------------------
    # Section 2: Process images functions from hedonic_alpha_images
    # for generating a GIF and community vs. resolution plots.
    # -----------------------------------------------------------
    print("\nRunning hedonic_alpha_images functions for a single image...")
    # Specify the input image and output folder for the GIF and plots.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg")):
            input_image = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            alpha_output_folder = os.path.join("results", "hedonic_gif")
            os.makedirs(alpha_output_folder, exist_ok=True)

            # 1) Load image and create superpixels
            original_image = hedonic_compare_segmentation.preprocess_image(input_image)
            segments = hedonic_compare_segmentation.create_superpixels(
                original_image, n_segments, compactness
            )
            # 2) Create graph from superpixels
            graph = hedonic_compare_segmentation.create_graph_from_superpixels(
                original_image, segments
            )
            density = graph.density()
            print(f"Graph density: {density:.4f}")

            # 3) Define resolution values (with denser sampling in the high resolution range)
            n_total = 100  # total frames
            threshold = 0.56
            n_low = int(n_total * 0.4)
            n_high = n_total - n_low
            res_low = np.linspace(0.0, threshold, n_low, endpoint=False)
            res_high = np.linspace(threshold, 1.0, n_high, endpoint=True)
            resolutions = np.concatenate((res_low, res_high))

            # 4) Generate a plot: resolution vs. number of communities
            alpha_values = []
            community_counts = []
            for res in resolutions:
                membership_global = hedonic_alpha_images.segment_graph_global_optimum(
                    graph, res
                )
                if membership_global is not None:
                    from collections import defaultdict

                    comm_dict = defaultdict(list)
                    for idx, comm in enumerate(membership_global):
                        comm_dict[comm].append(idx)
                    partition = list(comm_dict.values())
                else:
                    partition = hedonic_alpha_images.run_hedonic_multiple_times(
                        graph, res, runs=20
                    )
                alpha_values.append(res)
                community_counts.append(len(partition))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
            ax1.plot(
                alpha_values, community_counts, marker="o", markersize=4, linewidth=1
            )
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, max(community_counts) + 5)
            ax1.set_xlabel("Alpha (resolution)")
            ax1.set_ylabel("Number of Communities")
            ax1.set_title("Full Range")
            ax1.grid(True)
            ax2.plot(
                alpha_values, community_counts, marker="o", markersize=4, linewidth=1
            )
            ax2.set_xlim(0, 1)
            ax2.set_ylim(110, max(community_counts) + 5)
            ax2.set_xlabel("Alpha (resolution)")
            ax2.set_title("Zoom: Communities > 110")
            ax2.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(alpha_output_folder, "community_vs_alpha_zoom.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"Plot saved at: {plot_path}")

            # 5) Create segmentation frames and generate boomerang GIF
            frames = []
            for i, res in enumerate(resolutions):
                print(f"Segmenting with resolution = {res:.6f} ...")
                # Note: hedonic_alpha_images.segment_image returns (label_img, visual_img)
                label_img, visual_img = hedonic_alpha_images.segment_image(
                    graph, segments, original_image, res, runs=20
                )
                visual_img = hedonic_alpha_images.add_resolution_text(visual_img, res)
                frame_path = os.path.join(
                    alpha_output_folder, f"frame_{i:03d}_res_{res:.6f}.png"
                )
                cv2.imwrite(frame_path, cv2.cvtColor(visual_img, cv2.COLOR_RGB2BGR))
                frames.append(visual_img)
            boomerang_frames = frames + frames[-2::-1]
            gif_path = os.path.join(
                alpha_output_folder, f"{base_name}_hedonic_animation_boomerang.gif"
            )
            imageio.mimsave(gif_path, boomerang_frames, duration=0.5)
    print(f"GIF saved at: {gif_path}")

    print("\nCalculating segmentation metrics for fixed alpha values...")
    # For this example, the first available image is used
    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if image_files:
        test_image = os.path.join(input_folder, image_files[0])
        original_image = hedonic_compare_segmentation.preprocess_image(test_image)
        segments = hedonic_compare_segmentation.create_superpixels(
            original_image, n_segments, compactness
        )
        graph = hedonic_compare_segmentation.create_graph_from_superpixels(
            original_image, segments
        )

        # In this example, the superpixels (SLIC) are used as the reference (ground truth)
        ground_truth = segments

        test_alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
        metric_runs = 10
        metric_results = {"AMI": [], "Rand Index": [], "F1 Score": [], "IoU": []}
        metric_errors = {"AMI": [], "Rand Index": [], "F1 Score": [], "IoU": []}

        for alpha in test_alphas:
            ami_list, rand_list, f1_list, iou_list = [], [], [], []
            for _ in range(metric_runs):
                label_img, _ = hedonic_compare_metrics.segment_image(
                    graph, segments, original_image, alpha, runs=20
                )
                y_true = ground_truth.flatten()
                y_pred = label_img.flatten()
                ami_list.append(adjusted_mutual_info_score(y_true, y_pred))
                rand_list.append(rand_score(y_true, y_pred))
                f1_list.append(f1_score(y_true, y_pred, average="macro"))
                iou_list.append(jaccard_score(y_true, y_pred, average="macro"))
            metric_results["AMI"].append(np.mean(ami_list))
            metric_errors["AMI"].append(np.std(ami_list))
            metric_results["Rand Index"].append(np.mean(rand_list))
            metric_errors["Rand Index"].append(np.std(rand_list))
            metric_results["F1 Score"].append(np.mean(f1_list))
            metric_errors["F1 Score"].append(np.std(f1_list))
            metric_results["IoU"].append(np.mean(iou_list))
            metric_errors["IoU"].append(np.std(iou_list))

        x = np.arange(len(test_alphas))
        width = 0.18

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            x - 1.5 * width,
            metric_results["AMI"],
            width,
            yerr=metric_errors["AMI"],
            capsize=5,
            label="AMI",
            color="tab:blue",
        )
        ax.bar(
            x - 0.5 * width,
            metric_results["Rand Index"],
            width,
            yerr=metric_errors["Rand Index"],
            capsize=5,
            label="Rand Index",
            color="tab:orange",
        )
        ax.bar(
            x + 0.5 * width,
            metric_results["F1 Score"],
            width,
            yerr=metric_errors["F1 Score"],
            capsize=5,
            label="F1 Score",
            color="tab:green",
        )
        ax.bar(
            x + 1.5 * width,
            metric_results["IoU"],
            width,
            yerr=metric_errors["IoU"],
            capsize=5,
            label="IoU",
            color="tab:red",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{alpha:.1f}" for alpha in test_alphas])
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Metric Value")
        ax.set_title("Segmentation Metrics vs. Alpha (± SD)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        bar_metrics_path = os.path.join(output_folder, "metrics_vs_alpha.png")
        plt.savefig(bar_metrics_path)
        plt.show()
        print(f"Metrics plot saved at: {bar_metrics_path}")


if __name__ == "__main__":
    main()
