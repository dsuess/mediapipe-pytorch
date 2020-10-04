cc_library(
    name = "pytorch_inference_calculator",
    srcs = ["pytorch_inference_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:ret_check",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@com_google_absl//absl/memory",
        "@libtorch//:libtorch"
    ],
    alwayslink = 1,
)
