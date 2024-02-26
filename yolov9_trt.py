import os
import cv2
import argparse
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from python.AIResult import *
from python.logging_system import Logger
from python.tensorrt_base import TensorrtBase
from python.draw_AI_results import draw_detect_results
from python.decorators import time_cost, suppress_errors

parser = argparse.ArgumentParser("yolov9_demo")
parser.add_argument('--configs', type=str, default="configs", help="configs path")
parser.add_argument('--yaml_file', type=str, default="yolov9py.yaml", help="yaml file name")
parser.add_argument('--data', type=str, default="data", help="images data path")
args = parser.parse_args()


class Yolov9(TensorrtBase):
    def __init__(self, logger, config_path, config_file):
        super().__init__(logger)
        self.logger = logger
        assert self.read_parameters(config_path, config_file), self.logger.info("Read parameters failure!")
        assert self.create_engine_if_not_exit(), self.logger.error("create engine failure!")
        self.get_trt_model_stream()
    @time_cost
    @suppress_errors
    def preprocess(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    @time_cost
    def post_process(self, output, origin_h, origin_w):
        '''
        Post-process the output of YOLO model
        :param output: Output of model inference
        :param origin_h: Image original height
        :param origin_w: Image original width
        :return: Algorithm detection results
        '''
        predict = np.transpose(np.reshape(output, (self.output_dim, self.output_anchor_num)))
        detect_results = list()
        boxes_list = list()
        class_ids = list()
        scores = list()
        scores_array = np.max(predict[:, 4:], axis=1)
        filter_predict = predict[scores_array > self.conf_treshold, :]
        for predict_box in filter_predict:
            cx = predict_box[0]
            cy = predict_box[1]
            width = predict_box[2]
            height = predict_box[3]
            score = np.max(predict_box[4:])
            class_id = np.argmax(predict_box[4:])
            ratio_w = self.input_w / origin_w
            ratio_h = self.input_h / origin_h
            if ratio_h > ratio_w:
                left = (cx - width / 2) / ratio_w
                top = (cy - height / 2 - (self.input_h - ratio_w * origin_h) / 2) / ratio_w
                right = (cx + width / 2) / ratio_w
                bottom = (cy + height / 2 - (self.input_h - ratio_w * origin_h) / 2) / ratio_w
            else:
                left = (cx - width / 2 - (self.input_w - ratio_h * origin_w) / 2) / ratio_h
                top = (cy - height / 2) / ratio_h
                right = (cx + width / 2 - (self.input_w - ratio_h * origin_w) / 2) / ratio_h
                bottom = (cy + height / 2) / ratio_h
            box_xywh = list(map(lambda x: int(x), [max(0, left), max(0, top), min(right - left, origin_w), min(bottom - top, origin_h)]))
            boxes_list.append(box_xywh)
            class_ids.append(class_id)
            scores.append(score)
        nms_result = cv2.dnn.NMSBoxes(boxes_list, scores, self.conf_treshold, self.nms_threshold)
        for i in range(len(nms_result)):
            idx = nms_result[i]
            class_id = class_ids[idx]
            score = scores[idx]
            box = boxes_list[idx]
            result = DetResult(score, box, class_id)
            detect_results.append(result)
        return detect_results

    @time_cost
    @suppress_errors
    def do_infer(self, img):
        start_time = cv2.getTickCount()
        # Do image preprocess
        self.ctx.push()
        input_image, image_raw, h, w = self.preprocess(img)
        # Copy input image to host buffer
        np.copyto(self.host_inputs[0], input_image.ravel())
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.host_outputs[6], self.cuda_outputs[6], self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Here we use the first row of output in that batch_size = 1
        output = self.host_outputs[6]
        # Do postprocess
        detect_results = self.post_process(output, image_raw.shape[0], image_raw.shape[1])
        self.ctx.pop()
        # print cost time
        end_time = cv2.getTickCount()
        fps = 1 / ((end_time - start_time) / cv2.getTickFrequency())
        self.logger.info("detect fps:{}".format(fps))
        return detect_results

    @suppress_errors
    def get_trt_model_stream(self):
        '''
         Obtain the data flow for Tensorrt model inference and initialize the model
        '''
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = self.trt_logger
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(self.engine_file, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding_index, binding in enumerate(engine):
            self.logger.info("bingding shape:{}".format(engine.get_binding_shape(binding)))
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if binding_index == 0:
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            elif binding_index == 7:
                self.output_anchor_num = engine.get_binding_shape(binding)[-1]
                self.output_dim = engine.get_binding_shape(binding)[-2]
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings

    def read_parameters(self, config_path, config_file):
        '''
        Read parameters from config file
        :param config_path: Profile Path
        :param config_file: profile name
        :return: Did it read successfully
        '''
        yaml_file = os.path.join(config_path, config_file)
        if os.path.exists(yaml_file):
            fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
            self.conf_treshold = fs.getNode('confTreshold').real()
            self.nms_threshold = fs.getNode('nmsThreshold').real()
            self.quantization_infer = fs.getNode("quantizationInfer").string()
            self.onnx_file = os.path.join(config_path, fs.getNode('onnxFile').string())
            self.engine_file = os.path.join(config_path, fs.getNode('engineFile').string())
        else:
            return False
        return True

    def destroy(self):
        self.ctx.pop()
        del self.ctx
        self.logger.info("yolov9 destroy")


if __name__ == "__main__":
    log = Logger()
    logger = log.get_log("yolov9.txt")
    yolov9 = Yolov9(logger, args.configs, args.yaml_file)
    image_root = args.data
    file_list = os.listdir(image_root)
    for image_file in file_list:
        image_path = os.path.join(image_root, image_file)
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),  cv2.IMREAD_COLOR)
        detect_results = yolov9.do_infer(img)
        draw_detect_results(img, detect_results)
    yolov9.destroy()
