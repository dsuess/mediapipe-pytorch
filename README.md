# mediapipe-pytorch

This is a minimal example of how to build a [MediaPipe](https://github.com/google/mediapipe) component using pytorch for inference.
Currently, this only supports building a native C++/Desktop component -- no support for mobile platforms or JS. 

## Setup (TBC)

1. To build the example application, run:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 examples:pytorch_inference
```

2. Download the model file from [here](https://github.com/Nebula4869/YOLOv5-LibTorch/blob/master/yolov5s.torchscript.pt) to `examples/yolov5s.torchscript.pt. 
The path to the model is currently [hard-coded](https://github.com/dsuess/mediapipe-pytorch/blob/master/mediapipe_pytorch/pytorch_inference_calculator.cc#L56). 
Any torchscript model with the same input-output interface should work though.

3. To launch the application, run: 
```
GLOG_logtostderr=1 bazel-bin/examples/pytorch_inference
```


## TODO

[ ] Expand support to Android or iOS. The main issue here is that at the time of this writing, libtorch's bazel manifest only supports Desktop builds. Mobile bazel builds would require wrapping the cmake-based mobile build. 
