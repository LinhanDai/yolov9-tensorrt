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

#include <Yolov9.h>
#include "TimerCounter.h"

std::vector<cv::Scalar> generateRandomColors(int classNum)
{
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < classNum; i++)
    {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }
    return colors;
}

void showResult(const std::vector<detectResult>& result, std::vector<cv::Mat> &imgCloneBatch, std::vector<cv::Scalar> colors)
{
    for (int i = 0; i < result.size(); i++)
    {
        detectResult batchResult = result[i];
        for (const auto& r: batchResult)
        {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << "id:" << r.label << "  score:" << r.confidence;
            cv::rectangle(imgCloneBatch[i], cv::Point(r.left, r.top), cv::Point(r.right, r.bottom), colors[r.label], 2);
            cv::putText(imgCloneBatch[i], stream.str(), cv::Point(r.left, r.top - 5), 0, 0.8, colors[r.label], 2);
        }
        cv::imwrite("1.jpg", imgCloneBatch[i]);
        cv::namedWindow("Windows", cv::WINDOW_AUTOSIZE);
        cv::resizeWindow("Windows", imgCloneBatch[i].cols / 2, imgCloneBatch[i].rows / 2);
        cv::imshow("Windows", imgCloneBatch[i]);
        cv::waitKey(0);
    }
}

int main(int argc, char* argv[])
{
    std::string configPath = "../configs";
    std::string configFile = "yolov9.yaml";
    std::vector<cv::String> images;
    if (argc != 2)
    {
        std::cout << "Need input test img folder path!!!" << std::endl;
        return 0;
    }
    std::string folderPath = argv[1];
    cv::String path(folderPath + "/*.jpg");      //small picture
    cv::glob(path, images);
    std::shared_ptr<GPUTimer> timer = std::make_shared<GPUTimer>();
    std::shared_ptr<YoloV9> yoloObj = std::make_shared<YoloV9>(configPath, configFile);
    std::vector<cv::Scalar> colors = generateRandomColors(80);
    for (const auto& image: images)
    {
        std::vector<cv::Mat> imgMatVec;
        std::vector<unsigned char *> imgSrcVec;
        std::vector<ImgInfo> imgInfoVec;
        std::vector<detectResult> detectResult {};
        ImgInfo imgInfo{};
        cv::Mat img = cv::imread(image, cv::IMREAD_COLOR);
        imgInfo.width = img.cols;
        imgInfo.height = img.rows;
        imgInfo.channels = img.channels();
        unsigned char *deviceImgSrc;
        CHECK(cudaMalloc(&deviceImgSrc, img.cols * img.rows * img.channels() * sizeof(unsigned char)));
        CHECK(cudaMemcpy(deviceImgSrc, img.data, img.cols * img.rows * img.channels() * sizeof(unsigned char), cudaMemcpyHostToDevice));
        imgSrcVec.push_back(deviceImgSrc);
        imgInfoVec.push_back(imgInfo);
        imgMatVec.push_back(img);
        timer->start();
        yoloObj->doInfer(imgSrcVec, imgInfoVec, detectResult);
        timer->stop();
        float time = timer->elapsed_ms();
        std::cout << "cost time:" << time <<" ms, " <<"fps:" << 1000 / time << std::endl;
        showResult(detectResult, imgMatVec, colors);
        cudaFree(deviceImgSrc);
    }
    return 0;
}