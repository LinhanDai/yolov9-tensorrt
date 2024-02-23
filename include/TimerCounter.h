/**
* Website: https://github.com/LinhanDai
* @author dailinhan
* @date 24-02-23 9:30
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

#pragma once
#ifndef YOLOV9_TIMERCOUNTER_H
#define YOLOV9_TIMERCOUNTER_H

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

class CPUTimer
{
public:
    CPUTimer()
    {
        mStart = std::chrono::high_resolution_clock::now();
    }

    void start()
    {
        mStart = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        mEnd = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms()
    {
        int64_t dur = 0;
        dur = std::chrono::duration_cast<std::chrono::microseconds>(mEnd - mStart).count(); // us
        return (float)(dur) / 1000;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEnd;
};

class GPUTimer
{
public:
    GPUTimer()
    {
        cudaEventCreate(&mStart);
        cudaEventCreate(&mEnd);
    }

    float elapsed_ms()
    {
        float ms = 0;
        cudaEventElapsedTime(&ms, mStart, mEnd);
        return ms;
    }

    void start()
    {
        cudaEventRecord(mStart);
    }

    void stop()
    {
        cudaEventRecord(mEnd);
        cudaEventSynchronize(mEnd);
    }

private:
    cudaEvent_t mStart;
    cudaEvent_t mEnd;
};

#endif //YOLOV9_TIMERCOUNTER_H
