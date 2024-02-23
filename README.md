<center>
<h2> YOLOv9 tensorrt deployment </h2>
</center>

<h3>
This repository provides an API for accelerating inference deployment, with two open interface forms: C++and Python. C++also provides the use of CUDA programming to accelerate YOLOv9 model preprocessing and post-processing
</h3>

## Build

<h3> 1. Export onnx </h3>

Clone [YOLOv9](https://github.com/WongKinYiu/yolov9) code repository, download the original model provided by the repository, or train your own model, such as [yolov9-c.pt](https://objects.githubusercontent.com/github-production-release-asset-2e65be/759338070/c8ca43f2-0d2d-4aa3-a074-426505bfbfb1?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240223%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240223T073054Z&X-Amz-Expires=300&X-Amz-Signature=db76944695e398168b222b502bb019a301336e5b5dc74db31604699b8f837a9b&X-Amz-SignedHeaders=host&actor_id=45328395&key_id=0&repo_id=759338070&response-content-disposition=attachment%3B%20filename%3Dyolov9-c.pt&response-content-type=application%2Foctet-stream)

``` shell
# export onnx
python export.py --weights yolov9-c.pt --simplify --include "onnx"
```

<h3> 2. Setup </h3>

Place the exported onnx file in the "yolov9-tensorrt/configs" folder and configure the relevant parameters through the "yolov9-tensorrt/configs/yolov9.yaml" file
``` shell
# move onnx
cd yolov9-Tensorrt
mv yolov9-c.onnx ./configs
```

Modify parameter configuration in configs/yolov9-yaml
``` shell
# move onnx
cd yolov9-tensorrt
mv yolov9-c.onnx ./configs

# modify configuration in configs/yolov9.yaml
confTreshold: 0.25              #Detection confidence threshold
nmsTreshold : 0.45              #nms threshold
maxSupportBatchSize: 1          #support max input batch size
quantizationInfer: "FP16"       #support FP32 or FP16 quantization
onnxFile: "yolov9-c.onnx"       # The currently used onnx model file
engineFile: "yolov9-c.engine"   # Automatically generate file names for the Tensorrt inference engine
```

<h3> 3. Build project </h3>

``` shell
mkdir build
cd build
cmake ..
make -j4
```

## Run demo
The first run will generate the inference engine ".engine" file in the configs folder. If the inference engine has already been generated, it will not be generated again
``` shell
# run images floder
./demo ../data
```
<div align="center">

 ![图片](result/000000000036.jpg)
</div>

## 👏 Acknowledgement

This project is based on the following awesome projects:
- [Yolov9](https://github.com/WongKinYiu/yolov9) - YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information.
- [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - TensorRT samples and api documentation.
