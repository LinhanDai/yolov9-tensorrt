/**
* Website: https://github.com/LinhanDai
* @author dailinhan
* @date 24-02-23 10:24
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

#include <cuda_runtime.h>
#include <cstdio>


static __global__ void transpose_kernel(float *src, int num_bboxes, int num_elements,float *dst, int edge)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position>=edge)
        return;
    dst[position]=src[position / num_elements + (position % num_elements) * num_bboxes];
}

extern "C" void transpose_kernel_invoker(float *src, int num_bboxes, int num_elements,float *dst,cudaStream_t stream)
{
    int edge = num_bboxes * num_elements;
    int block = 256;
    int gird = ceil(edge / (float)block);
    transpose_kernel<<<gird,block,0,stream>>>(src,num_bboxes,num_elements, dst, edge);
}

static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(
        float* predict, int num_bboxes, int num_classes,
        float confidence_threshold, float* invert_affine_matrix,
        float* parray, int max_objects, int NUM_BOX_ELEMENT)
{

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (4 + num_classes) * position;

    float* class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if(*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }

    // confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;
    // printf("index %d max_objects %d\n", index,max_objects);
    float cx         = pitem[0];
    float cy         = pitem[1];
    float width      = pitem[2];
    float height     = pitem[3];

    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);


    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
        float aleft, float atop, float aright, float abottom,
        float bleft, float btop, float bright, float bbottom)
{
    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes,
                                       int max_objects,
                                       float threshold,
                                       int NUM_BOX_ELEMENT)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i)
    {
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4])
        {
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                    pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                    pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold)
            {
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

extern "C" void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold,
        float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects,
        int num_box_element, cudaStream_t stream)
{
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;
    decode_kernel<<<grid, block, 0, stream>>>(
            predict, num_bboxes, num_classes,
            confidence_threshold, invert_affine_matrix,
            parray, max_objects, num_box_element);

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, num_box_element);
}