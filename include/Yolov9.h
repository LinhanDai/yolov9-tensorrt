/**
* Website: https://github.com/LinhanDai
* @author dailinhan
* @date 24-02-23 9:19
                   _ooOoo_
                  o8888888o
                  88" . "88
                  (| -_- |)
                  O\  =  /O
               ____/`---'\____
             .'  \\|     |//  `.
            /  \\|||  :  |||//  \
           /  _||||| -:- |||||-  \
           |   | \\\  -  /// |   |
           | \_|  ''\---/''  |   |
           \  .-\__  `-`  ___/-. /
         ___`. .'  /--.--\  `. . __
      ."" '<  `.___\_<|>_/___.'  >'"".
     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
     \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            no error       no bug
*/

#ifndef YOLOV9_TENSORRT_YOLOV9_H
#define YOLOV9_TENSORRT_YOLOV9_H

#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <exception>
#include <NvInfer.h>
#include <memory>
#include <cuda_runtime.h>
#include <NvOnnxParser.h>
#include "LoggingRT.h"


#define CHECK(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}

#define MAX_OBJECTS 1000
#define NUM_BOX_ELEMENT 7   // left, top, right, bottom, confidence, class, keepflag
#define GPU_MAX_LIMIT_WIDTH 4096
#define GPU_MAX_LIMIT_HEIGHT 4096
#define GPU_MAX_LIMIT_CHANNEL 3

inline bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if(code != cudaSuccess)
    {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        std::cout << "runtime error " << file << ":" << line << " :" << "  " << op << " failed, code:" << err_name << " massage:" << err_message << std::endl;
        return false;
    }
    return true;
}

struct ImgInfo
{
    int width;
    int height;
    int channels;
};

struct Box{
    int left, top, right, bottom;
    float confidence;
    int label;
    int trackerID;

    Box() = default;
    Box(int left, int top, int right, int bottom, float confidence, int label, int trackerID):
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label),trackerID(trackerID){}
};
typedef std::vector<Box> detectResult;


extern "C" void transpose_kernel_invoker(float *src, int num_bboxes, int num_elements,float *dst,cudaStream_t stream);

extern "C" void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold,
        float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects,
        int num_box_element, cudaStream_t stream);

extern "C"  void preprocess_kernel_img(
        uint8_t* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
        float *d2i, cudaStream_t stream);

class YoloV9
{
public:
    struct AffineMatrix  //Preprocessing affine transformation matrix and inverse matrix
    {
        float i2d[6];   //transformation matrix
        float d2i[6];   //inverse matrix
    };

public:
    explicit YoloV9(const std::string& configPath, const std::string &configFile);
    void doInfer(std::vector<unsigned char *> batchImg,
                 std::vector<ImgInfo> imgInfoVec,
                 std::vector<detectResult> &detResult);

private:
    std::vector<detectResult> getDetResultToCPU(int batch);
    void getAffineMartrix(AffineMatrix &afmt,cv::Size &to,cv::Size &from);
    void gpuDecode(float* anchorsProb, int batch, float confidence_threshold, float nms_threshold);
    void imgPreProcess(std::vector<unsigned char *> &batchImg);
    void getTrtmodelStream();
    void getBindingDimsInfo();
    void createInferenceEngine(nvinfer1::IHostMemory **modelStream);;
    void modelInfer(nvinfer1::IExecutionContext& context, int batchSize);
    bool readParameters(const std::string& configPath, const std::string& configFile);
    bool createEngineIfNotExit();
    nvinfer1::IHostMemory *createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config);

private:
    int mMaxSupportBatchSize{};
    int mInputH{};
    int mInputW{};
    int mInputC{};
    int mOutputAnchorsNum;
    int mOutputAnchorsDim;
    int mOutputAnchorsSize;
    std::string mOnnxFile;
    std::string mEngineFile;
    std::string mQuantizationInfer;
    unsigned char *mDeviceWarpAffine;
    char *mTrtModelStream{};
    nvinfer1::IRuntime *mRuntime{};
    nvinfer1::ICudaEngine *mEngine{};
    nvinfer1::IExecutionContext *mContext{};
    cudaStream_t  mStream{};
    float *mAffineMatrixD2iHost;
    float *mAffineMatrixD2iDevice;
    float mConfTreshold;
    float mNMSTreshold;
    float *mBuff[9];
    float* mOutputDevice;
    float* mTransposeDevice;
    float* mOutputHost;
    std::vector<ImgInfo> mImageSizeBatch;
    Logger mLogger;
};

#endif //YOLOV9_TENSORRT_YOLOV9_H
