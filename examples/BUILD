package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "add_detections",
    deps = [
        "@mediapipe//mediapipe/examples/desktop:demo_run_graph_main",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe_addons:add_detections_calculator",
    ],
)
