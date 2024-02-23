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

#include "Yolov9.h"
#include "TimerCounter.h"


YoloV9::YoloV9(const std::string& configPath, const std::string &configFile)
{
    std::cout << "Yolov9 init..." << std::endl;
    assert(readParameters(configPath, configFile));
    cudaSetDevice(0);
    assert(createEngineIfNotExit() == true && "engine create failure!");
    getTrtmodelStream();
}

nvinfer1::IHostMemory *YoloV9::createEngine(nvinfer1::IBuilder *builder, nvinfer1::IBuilderConfig *config)
{
    std::cout << "Creating an inference engine, please wait a few minutes!!!" << std::endl;
    mLogger.setReportableSeverity(Severity::kERROR);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(explicitBatch);
    assert(network);
    nvonnxparser::IParser *parser = nvonnxparser::createParser(*network, mLogger);
    assert(parser);
    bool parsed = parser->parseFromFile(mOnnxFile.c_str(), (int) nvinfer1::ILogger::Severity::kWARNING);
    if (!parsed) {
        mLogger.logPrint(Severity::kERROR, __FUNCTION__ , __LINE__, "onnx file parse error, please check onnx file!");
        std::abort();
    }
    config->setMaxWorkspaceSize(2_GiB);
    if (strcmp(mQuantizationInfer.c_str(), "FP16") == 0)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if(strcmp(mQuantizationInfer.c_str(), "FP32") == 0)
    {
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }
    nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
    if (inputDims.d[0] == -1)
    {
        nvinfer1::IOptimizationProfile *profileCalib = builder->createOptimizationProfile();
        const auto inputName = network->getInput(0)->getName();
        nvinfer1::Dims batchDim = inputDims;
        batchDim.d[0] = 1;
        // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, batchDim);
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, batchDim);
        batchDim.d[0] = mMaxSupportBatchSize;
        profileCalib->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, batchDim);
        config->addOptimizationProfile(profileCalib);
    }
    nvinfer1::IHostMemory* serialized_model = builder->buildSerializedNetwork(*network, *config);
    assert(serialized_model);
    mLogger.logPrint(Severity::kINFO,__FUNCTION__ ,__LINE__ ,"success create serialized_model!");
    return serialized_model;
}

void YoloV9::createInferenceEngine(nvinfer1::IHostMemory **modelStream)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(mLogger);
    assert(builder);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    assert(config);
    (*modelStream) = createEngine(builder, config);
    assert(modelStream != nullptr && "engine create failure!");
}

bool YoloV9::createEngineIfNotExit()
{
    std::ifstream cache(mEngineFile.c_str(), std::ios::binary);
    if (cache)
        return true;
    else {
        nvinfer1::IHostMemory *modelStream{nullptr};
        createInferenceEngine(&modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(mEngineFile.c_str(), std::ios::binary);
        if (!p) {
            std::cout << "could not open plan output file" << std::endl;
            return false;
        }
        p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
    }
}

bool YoloV9::readParameters(const std::string& configPath, const std::string& configFile)
{
    std::string yamlFile = configPath + "/" + configFile;
    if (access(yamlFile.c_str(), F_OK) != -1)
    {
        cv::FileStorage fs(yamlFile, cv::FileStorage::READ);
        mConfTreshold = fs["confTreshold"];
        mNMSTreshold = fs["nmsTreshold"];
        mMaxSupportBatchSize = fs["maxSupportBatchSize"];
        mQuantizationInfer = (std::string) fs["quantizationInfer"];
        mOnnxFile = configPath + "/"  + (std::string) fs["onnxFile"];
        mEngineFile = configPath + "/" + (std::string) fs["engineFile"];
    }
    else
    {
        return false;
    }
    return true;
}

void YoloV9::getBindingDimsInfo()
{
    nvinfer1::Dims inputDims = mEngine->getBindingDimensions(0);
    nvinfer1::Dims dInput = inputDims;
    mInputC = dInput.d[1];
    mInputH = dInput.d[2];
    mInputW = dInput.d[3];
    nvinfer1::Dims outPutBoxesDims = mEngine->getBindingDimensions(7);
    nvinfer1::Dims dOutPutBoxes = outPutBoxesDims;
    mOutputAnchorsDim= dOutPutBoxes.d[1];
    mOutputAnchorsNum = dOutPutBoxes.d[2];
    mOutputAnchorsSize = mOutputAnchorsNum * mOutputAnchorsDim;
}

void YoloV9::getTrtmodelStream()
{
    int engineFileSize = 0;
    cudaSetDevice(0);
    std::ifstream file(mEngineFile, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        engineFileSize = file.tellg();
        file.seekg(0, file.beg);
        mTrtModelStream = new char[engineFileSize];
        assert(mTrtModelStream);
        file.read(mTrtModelStream, engineFileSize);
        file.close();
    }
    mRuntime = nvinfer1::createInferRuntime(mLogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(mTrtModelStream, engineFileSize);
    assert(mEngine);
    mContext = mEngine->createExecutionContext();
    assert(mContext);
    getBindingDimsInfo();
    //create fixed maximum input buffer
    int inputSingleByteNum = mInputW * mInputH * mInputC;
    int outputSingleAnchorByteNum = mOutputAnchorsNum * mOutputAnchorsDim;
    //input layer
    CHECK(cudaMalloc(&(mBuff[0]), mMaxSupportBatchSize * inputSingleByteNum * sizeof(float)));
    //output feature map layer
    nvinfer1::Dims outputDims1 = mEngine->getBindingDimensions(1);
    CHECK(cudaMalloc(&(mBuff[1]), mMaxSupportBatchSize * outputDims1.d[1] * outputDims1.d[2] * outputDims1.d[3] * sizeof(float)));
    nvinfer1::Dims outputDims2 = mEngine->getBindingDimensions(2);
    CHECK(cudaMalloc(&(mBuff[2]), mMaxSupportBatchSize * outputDims2.d[1] * outputDims2.d[2] * outputDims2.d[3] * sizeof(float)));
    nvinfer1::Dims outputDims3 = mEngine->getBindingDimensions(3);
    CHECK(cudaMalloc(&(mBuff[3]), mMaxSupportBatchSize * outputDims3.d[1] * outputDims3.d[2] * outputDims3.d[3] * sizeof(float)));
    nvinfer1::Dims outputDims4 = mEngine->getBindingDimensions(4);
    CHECK(cudaMalloc(&(mBuff[4]), mMaxSupportBatchSize * outputDims4.d[1] * outputDims4.d[2] * outputDims4.d[3] * sizeof(float)));
    nvinfer1::Dims outputDims5 = mEngine->getBindingDimensions(5);
    CHECK(cudaMalloc(&(mBuff[5]), mMaxSupportBatchSize * outputDims5.d[1] * outputDims5.d[2] * outputDims5.d[3] * sizeof(float)));
    nvinfer1::Dims outputDims6 = mEngine->getBindingDimensions(6);
    CHECK(cudaMalloc(&(mBuff[6]), mMaxSupportBatchSize * outputDims6.d[1] * outputDims6.d[2] * outputDims6.d[3] * sizeof(float)));
    //output layer
    CHECK(cudaMalloc(&(mBuff[7]), mMaxSupportBatchSize * outputSingleAnchorByteNum * sizeof(float)));
    CHECK(cudaMalloc(&(mBuff[8]), mMaxSupportBatchSize * outputSingleAnchorByteNum * sizeof(float)));

    //malloc resize warpAffine space
    mDeviceWarpAffine = nullptr;
    CHECK(cudaMalloc(&mDeviceWarpAffine, GPU_MAX_LIMIT_WIDTH * GPU_MAX_LIMIT_HEIGHT * GPU_MAX_LIMIT_CHANNEL * sizeof(unsigned char)));
    CHECK(cudaMemset(mDeviceWarpAffine, 0, GPU_MAX_LIMIT_WIDTH * GPU_MAX_LIMIT_HEIGHT * GPU_MAX_LIMIT_CHANNEL * sizeof(unsigned char)));

    //malloc yolo gpuDecode space
    mOutputDevice = nullptr;
    mTransposeDevice = nullptr;
    mOutputHost = nullptr;
    CHECK(cudaMalloc(&mOutputDevice, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));
    CHECK(cudaMalloc(&mTransposeDevice, mOutputAnchorsSize * sizeof(float)));
    CHECK(cudaMallocHost(&mOutputHost, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));
    CHECK(cudaMemset(mOutputHost, 0, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));
    CHECK(cudaMemset(mTransposeDevice, 0, mOutputAnchorsSize * sizeof(float)));
    CHECK(cudaMemset(mOutputDevice, 0, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));

    mAffineMatrixD2iHost = nullptr;
    mAffineMatrixD2iDevice = nullptr;
    CHECK(cudaMallocHost(&mAffineMatrixD2iHost,sizeof(float) * 6));
    CHECK(cudaMalloc(&mAffineMatrixD2iDevice,sizeof(float) * 6));
    delete []mTrtModelStream;
    mTrtModelStream = nullptr;
}

void YoloV9::getAffineMartrix(AffineMatrix &afmt,cv::Size &to,cv::Size &from)
{
    float scale = std::min(to.width/(float)from.width,to.height/(float)from.height);
    afmt.i2d[0] = scale;
    afmt.i2d[1] = 0;
    afmt.i2d[2] = (-scale * from.width+to.width) * 0.5;
    afmt.i2d[3] = 0;
    afmt.i2d[4] = scale;
    afmt.i2d[5] = (-scale * from.height + to.height) * 0.5;
    cv::Mat  cv_i2d(2,3,CV_32F,afmt.i2d);
    cv::Mat  cv_d2i(2,3,CV_32F,afmt.d2i);
    cv::invertAffineTransform(cv_i2d,cv_d2i);
    memcpy(afmt.d2i,cv_d2i.ptr<float>(0),sizeof(afmt.d2i));
}

void YoloV9::imgPreProcess(std::vector<unsigned char *> &batchImg)
{
    for (size_t i = 0; i < batchImg.size(); i++)
    {
        AffineMatrix afmt{};
        cv::Size to(mInputW, mInputH);
        cv::Size from(mImageSizeBatch[i].width, mImageSizeBatch[i].height);
        getAffineMartrix(afmt, to, from);
        memcpy(mAffineMatrixD2iHost,afmt.d2i,sizeof(afmt.d2i));
        CHECK(cudaMemcpyAsync(mAffineMatrixD2iDevice, mAffineMatrixD2iHost, sizeof(afmt.d2i),cudaMemcpyHostToDevice, mStream));
        preprocess_kernel_img(batchImg[i], mImageSizeBatch[i].width, mImageSizeBatch[i].height,
                                     mBuff[0], mInputW, mInputH, mAffineMatrixD2iDevice, mStream);
    }
}

void YoloV9::gpuDecode(float* anchorsProb, int batch, float confidence_threshold, float nms_threshold)
{
    for (int i = 0; i < batch; i++)
    {
        float *predictDevice = anchorsProb + i * mOutputAnchorsSize;
        transpose_kernel_invoker(predictDevice, mOutputAnchorsNum, mOutputAnchorsDim, mTransposeDevice, mStream);
        CHECK(cudaMemset(mOutputDevice, 0, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));
        decode_kernel_invoker(
                mTransposeDevice, mOutputAnchorsNum,
                mOutputAnchorsDim - 4, confidence_threshold,
                nms_threshold, mAffineMatrixD2iDevice,
                mOutputDevice, MAX_OBJECTS,
                NUM_BOX_ELEMENT,mStream);
    }
}

std::vector<detectResult> YoloV9::getDetResultToCPU(int batch)
{
    std::vector<detectResult> result;
    for (int b = 0; b < batch; b++)
    {
        std::vector<Box> boxResult;
        CHECK(cudaMemset(mOutputHost, 0, sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float)));
        CHECK(cudaMemcpyAsync(mOutputHost, mOutputDevice,
                              sizeof(float) + MAX_OBJECTS * NUM_BOX_ELEMENT * sizeof(float),
                              cudaMemcpyDeviceToHost, mStream));
        CHECK(cudaStreamSynchronize(mStream));
        int num_boxes = std::min((int)mOutputHost[0], MAX_OBJECTS);
        for(int i = 0; i < num_boxes; ++i)
        {
            float* ptr = mOutputHost + 1 + NUM_BOX_ELEMENT * i;
            int keep_flag = ptr[6];
            if(keep_flag)
            {
                boxResult.emplace_back(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5], 0);
            }
        }
        result.push_back(boxResult);
    }
    return result;
}

void YoloV9::modelInfer(nvinfer1::IExecutionContext& context, int batchSize)
{
    const nvinfer1::ICudaEngine &engine = context.getEngine();
    nvinfer1::Dims inputDims = engine.getBindingDimensions(0);
    nvinfer1::Dims d = inputDims;
    d.d[0] = batchSize;
    if (!mContext->setBindingDimensions(0, d))
    {
        mLogger.logPrint(Severity::kERROR, __FUNCTION__ , __LINE__, "The input dimension of the model is incorrect");
        std::abort();
    }
    context.enqueueV2((void **)mBuff, mStream, nullptr);
}

void YoloV9::doInfer(std::vector<unsigned char *> batchImg,
                     std::vector<ImgInfo> imgInfoVec,
                     std::vector<detectResult> &detResult)
{
    int batch = imgInfoVec.size();
    mImageSizeBatch = imgInfoVec;
    imgPreProcess(batchImg);
    modelInfer(*mContext, batch);
    gpuDecode(mBuff[7], batch,mConfTreshold, mNMSTreshold);
    detResult = getDetResultToCPU(batch);
}