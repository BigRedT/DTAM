// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.


#ifndef COSTVOLUME_HPP
#define COSTVOLUME_HPP


#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include "CostVolume.cuh"
typedef  int FrameID;

class CostVolume
{
    
public:
    FrameID fid;
    int rows;
    int cols;
    int layers;
    float near; //inverse depth of center of voxels in layer layers-1
    float far;  //inverse depth of center of voxels in layer 0
    float depthStep;
    float initialWeight;
    cv::Mat R;
    cv::Mat T;
    cv::Mat cameraMatrix;//Note! should be in OpenCV format
    float occlusionThreshold;
    Norm norm;
    
    cv::Mat projection;//projects world coordinates (x,y,z) into (rows,cols,layers)

    cv::gpu::GpuMat baseImage;
    cv::gpu::GpuMat baseImageGray;
    cv::gpu::GpuMat lo;
    cv::gpu::GpuMat hi;
    cv::gpu::GpuMat loInd;

    float * data;
    float * hits;

    cv::gpu::GpuMat dataContainer;
    cv::gpu::GpuMat hitContainer;

    int count;
    cv::gpu::Stream cvStream;

    void updateCost(const cv::Mat& image, const cv::Mat& R, const cv::Mat& T);//Accepts pinned RGBA8888 or BGRA8888 for high speed
    
    CostVolume(){}
    ~CostVolume();
    CostVolume(cv::Mat image, FrameID _fid, int _layers, float _near, float _far,
            cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix, float occlusionThreshold,
            Norm norm, float initialCost=3.0, float initialWeight=.001);

    //HACK: remove this function in release
    cv::Mat downloadOldStyle( int layer){
        cv::Mat cost;
        cv::gpu::GpuMat tmp=dataContainer.rowRange(layer,layer+1);
        tmp.download(cost);
        cost=cost.reshape(0,rows);
        return cost;
    }

    std::vector<cv::Mat> download( ){
        cv::Mat out;
        std::vector<cv::Mat>  all;
        dataContainer.download(out);
        out=out.t();
        out=out.reshape(0,rows);
        for(int i=0;i<rows;i++){
            all.push_back(out.rowRange(i,i+1).clone().reshape(0,cols));
        }
        return all;
    }
private:
    void solveProjection(const cv::Mat& R, const cv::Mat& T);
    void checkInputs(const cv::Mat& R, const cv::Mat& T,
            const cv::Mat& _cameraMatrix);
    void simpleTex(const cv::Mat& image,cv::gpu::Stream cvStream=cv::gpu::Stream::Null());

private:
    //temp variables ("static" containers)
    cv::Ptr<char> _cuArray;//Ptr<cudaArray*> really
    cv::Ptr<char> _texObj;//Ptr<cudaTextureObject_t> really
    cv::Ptr<char> _cuArray2;//Ptr<cudaArray*> really
    cv::Ptr<char> _texObj2;//Ptr<cudaTextureObject_t> really
    cv::Mat cBuffer;//Must be pagable
    cv::Ptr<char> ref;
    
    struct m34c{//for storing the old perspective matrix
            float data[12];
        };
    m34c p2;
};

#endif // COSTVOLUME_HPP
