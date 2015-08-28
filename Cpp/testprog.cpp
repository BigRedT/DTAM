#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>


//Mine
#include "fileLoader.hpp"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
// #include "OpenDTAM.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "Track/Track.hpp"

#include "utils/utils.hpp"


//debug
#include "tictoc.h"


namespace patch
{
  template < typename T > std::string to_string( const T& n )
  {
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
  }
}

const static bool valgrind=0;

//A test program to make the mapper run
using namespace cv;
using namespace cv::gpu;
using namespace std;

int App_main( int argc, char** argv );

void myExit(){
    ImplThread::stopAllThreads();
}
int main( int argc, char** argv ){

    initGui();

    int ret=App_main(argc, argv);
    myExit();
    return ret;
}


int App_main( int argc, char** argv )
{
    volatile int debug=0; 
    srand(314159);
    rand();
    rand();
    cv::theRNG().state = rand();
    
    int numImg=100;

#if !defined WIN32 && !defined _WIN32 && !defined WINCE && defined __linux__ && !defined ANDROID
    pthread_setname_np(pthread_self(),"App_main");
#endif

    char filename[500];

    std::string pathDepthDir = "/home/tgupta6/OpenDTAM/Cpp/Build/depthMaps";
    boost::filesystem::create_directory(pathDepthDir);

    Mat image, cameraMatrix, R, T;
    vector<Mat> images,Rs,ds,Ts,Rs0,Ts0,D0;
    
    Mat ret;//a place to return downloaded images to
    
    double reconstructionScale=1;
    int inc=1;
    for(int i=0;i<numImg;i+=inc){
        Mat tmp,d,image;
        int offset=0;
          
        loadAhanda("/home/tgupta6/OpenDTAM/Trajectory_30_seconds/",
                   65535,
                   i+offset,
                   image,
                   d,
                   cameraMatrix,
                   R,
                   T);
        tmp=Mat::zeros(image.rows,image.cols,CV_32FC3);
        randu(tmp,0,1);
        resize(image+tmp/255,image,Size(),reconstructionScale,reconstructionScale);
        images.push_back(image.clone());
        Rs.push_back(R.clone());
        Ts.push_back(T.clone());
        ds.push_back(d.clone());
        Rs0.push_back(R.clone());
        Ts0.push_back(T.clone());
        D0.push_back(1/d);
    
        
    }
    
    CudaMem cret(images[0].rows,images[0].cols,CV_32FC1);
    ret=cret.createMatHeader();
    
    //Setup camera matrix
    double sx=reconstructionScale;
    double sy=reconstructionScale;
    
    
    int layers=64;
    int imagesPerCV=10;
    float occlusionThreshold=.05;
    Norm norm=L1T;

    for(int startAt=imagesPerCV; startAt < numImg-imagesPerCV; ++startAt) {
      CostVolume cv(images[startAt],(FrameID)startAt,layers,0.015,0.0,Rs[startAt],Ts[startAt],cameraMatrix,occlusionThreshold,norm);
    
  
      cv::gpu::Stream s;
      double totalscale=1.0;
      int tcount=0;
      int sincefail=0;
 
      for(int imageNum=startAt-imagesPerCV; imageNum<startAt+imagesPerCV; imageNum++) {
	if(imageNum == startAt) { continue; }
      
	cout<<"using: "<< imageNum<<endl;
	cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
	cudaDeviceSynchronize();
      }      
    
      //Attach optimizer
      Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(cv.baseImageGray,cv.cvStream);
      DepthmapDenoiseWeightedHuber& denoiser=*dp;
      Optimizer optimizer(cv);
      optimizer.initOptimization();
      GpuMat a(cv.loInd.size(),cv.loInd.type());
      cv.cvStream.enqueueCopy(cv.loInd,a);
      GpuMat d;
      denoiser.cacheGValues();
      ret=image*0;
            
      bool doneOptimizing; 
      do{
	a.download(ret);
	pfShow("A function", ret, 0, cv::Vec2d(0, layers));
	for (int i = 0; i < 10; i++) {
	  d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
	}
	doneOptimizing=optimizer.optimizeA(d,a);
	std::cout << "Optimizing..." << std::endl;
      }while(!doneOptimizing);
    
      optimizer.cvStream.waitForCompletion();
      cv::Mat depthMap = optimizer.depthMap();
      double min, max;
      cv::minMaxLoc(depthMap, &min, &max);
      std::cout << "Max: " << max << " Min: " << min << std::endl;
      cv::Mat depthImg;
      depthMap.convertTo(depthImg, CV_8U, 255/max, 0);
      medianBlur(depthImg,depthImg,3);
      // cv::namedWindow("depthmap",1);
      // cv::imshow("depthmap",depthImg);
      // cv::waitKey();
      std::string depthMapNum;
      //      std::sprintf(depthMapNum, "%d", startAt);
      cv::imwrite(pathDepthDir + '/' + patch::to_string(startAt) + ".jpg", depthImg);
    }
    
    Stream::Null().waitForCompletion();
    return 0;
}


