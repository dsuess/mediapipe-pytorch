package(default_visibility = ["//examples:__subpackages__"])

cc_binary(
    name = "pytorch_inference",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "@mediapipe//mediapipe/calculators/util:non_max_suppression_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe_pytorch:pytorch_inference_calculator",
    ],
)
