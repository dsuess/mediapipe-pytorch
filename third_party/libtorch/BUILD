package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "osx",
    constraint_values = [
        "@platforms//os:osx",
    ],
)

cc_library(
    name = "libtorch",
    deps = [
        ":torch",
        ":caffe2"
    ]
)

cc_library(
    name = 'torch',
    hdrs = glob(
        [
            'include/torch/**/*.h',
        ], exclude = [
            'include/torch/csrc/api/include/**/*.h'
        ]
    ) + glob([
        'include/torch/csrc/api/include/**/*.h'
    ]),
    srcs = select({
        ":windows": [
            'lib/torch.lib',
            'lib/torch_cuda.lib',
            'lib/torch_cpu.lib',
            'lib/torch_global_deps.dll',
        ],
        ":osx": [
            'lib/libtorch.dylib',
            'lib/libtorch_cpu.dylib',
            'lib/libtorch_global_deps.dylib',
        ],
        "//conditions:default": [
            'lib/libtorch.so',
            'lib/libtorch_cuda.so',
            'lib/libtorch_cpu.so',
            'lib/libtorch_global_deps.so',
        ],
    }),
    deps = [
        ":ATen",
        #":c10_cuda",
        ":c10",
    ],
    includes = [
        "include",
        "include/torch/csrc/api/include/"
    ]
)

cc_library(
    name = 'c10_cuda',
    hdrs = glob([
        'include/c10/**/*.h'
    ]),
    srcs = select({
        ":windows": ["lib/c10_cuda.lib"],
        ":osx": ["lib/libc10.dylib"],
        "//conditions:default": ["lib/libc10_cuda.so"],
    }),
    strip_include_prefix = "include",
    deps = [
        ":c10"
    ]
)

cc_library(
    name = 'c10',
    hdrs = glob([
        'include/c10/**/*.h'
    ]),
    srcs = select({
        ":windows": ["lib/c10.lib"],
        ":osx": ["lib/libc10.dylib"],
        "//conditions:default": ["lib/libc10.so"],
    }),
    strip_include_prefix = "include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "include/ATen/**/*.h"
    ]),
    strip_include_prefix = "include"
)

cc_library(
    name = 'caffe2',
    hdrs = glob([
        'include/caffe2/**/*.h'
    ]),
    srcs = select({
        ":windows": [
            'lib/caffe2_nvrtc.lib',
            'lib/caffe2_detectron_ops_gpu.lib',
            'lib/caffe2_module_test_dynamic.lib'
        ],
        ":osx": [
            'lib/libcaffe2_detectron_ops.dylib',
            'lib/libcaffe2_observers.dylib',
            'lib/libcaffe2_module_test_dynamic.dylib'
        ],
        "//conditions:default": [
            'lib/libcaffe2_nvrtc.so',
            'lib/libcaffe2_detectron_ops_gpu.so',
            'lib/libcaffe2_observers.so',
            'lib/libcaffe2_module_test_dynamic.so'
        ],
    }),
    strip_include_prefix = "include",
)
