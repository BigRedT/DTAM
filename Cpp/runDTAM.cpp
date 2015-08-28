#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>


//OpenDTAM
#include "fileLoader.hpp"
#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
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

// Initialize Camera Parameters
/*
const double fx = 2284.80;
const double fy = 2701.04;
const double cx = 954.65;
const double cy = 530.48;
const double k1 = 0.0638;
const double k2 = 0.2947;
const double k3 = -1.6394;
const double p1 = -0.0009;
const double p2 = -0.0014;
*/

const double fx = 1745.75/2;
const double fy = 1760.71/2;
const double cx = 997.65/2;
const double cy = 558.68/2;
const double k1 = 0.0660;
const double k2 = 0.2850;
const double k3 = -1.6434;
const double p1 = -0.0008;
const double p2 = -0.0015;

const int NUM_FRAMES = 50;
const int FRAME_WIDTH = 1920/2;
const int FRAME_HEIGHT = 1088/2;

void undistortFrame(const cv::Mat &frame, cv::Mat &undistortedFrame) {
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat distCoeffs = (cv::Mat_<double>(5,1) << k1, k2, p1, p2, k3);
  cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);
}

void readVideo(const std::string &videoName, std::vector<cv::Mat> &images) {
  // open video file to be read
  cv::VideoCapture videoObj;
  videoObj.open(videoName);
  if(!videoObj.isOpened()) {
    std::cout << "The video file could not be openned" << std::endl;
    return;
  }
  else {
    std::cout << "The video file was succesfully opened" << std::endl;
  }
  
  // cv::namedWindow("ReadVideo",1);
  
  cv::Mat _frame, __frame, __frame_resized, frame;
  videoObj >> _frame;
  if(_frame.empty()) {
    std::cout << "No frames to process" << std::endl;
    return;
  }
  //  cv::cvtColor(_frame, __frame, CV_BGR2GRAY);
  cv::resize(_frame, __frame_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
  undistortFrame(__frame_resized, frame);
  frame.convertTo(__frame, CV_32FC3, 1/255.0, 1/255.0);
  images.push_back(__frame);
  
  //  cv::imshow("ReadVideo",frame);
  //cv::waitKey(1);

  int count = 1;
  while(count < NUM_FRAMES) {
    std::cout << count << std::endl;
    count++;
    videoObj >> _frame;
    if(_frame.empty()) {
      std::cout << "Processed entire video" << std::endl;
      return;
    }
    //    cv::cvtColor(_frame, __frame, CV_BGR2GRAY);
    cv::resize(_frame, __frame_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
    undistortFrame(__frame_resized, frame);
    frame.convertTo(__frame, CV_32FC3, 1/255.0, 1/255.0);
    images.push_back(__frame);
    //cv::imshow("ReadVideo",frame);
    //cv::waitKey();
  }
}

bool readCameraExtrinsics(const int &numImg, 
			  const std::string &cameraExtrinsicsTxt,
			  std::vector<cv::Mat> &Rs,
			  std::vector<cv::Mat> &Ts) {
  
  std::ifstream cameraExtTxt(cameraExtrinsicsTxt.c_str());
  if(!cameraExtTxt) {
    std::cout << "The file could not be opened" << std::endl;
    return false;
  }
  
  // Read the file header : Number of Camera Poses
  int numPoses, frameNum;
  cameraExtTxt >> numPoses; 
  
  cv::Mat R(3,3,CV_64FC1), T(3,1,CV_64FC1);
  float val;
  int counter=0;
  while(cameraExtTxt && counter < numPoses) {
    // read the frame number
    cameraExtTxt >> frameNum;
    std::cout << "Frame Number: " << frameNum << std::endl;

    // Read Translation
    for(int i=0; i < 3; ++i) {
      cameraExtTxt >> val;
      T.at<double>(i,0) = val;
    }
    std::cout << T << std::endl;
    
    
    // Read Rotation
    for(int i=0; i < 3; ++i) {
      for(int j=0; j < 3; ++j) {
	cameraExtTxt >> val;
	R.at<double>(j,i) = val;
      }
    }
    std::cout << R << std::endl;
    
    Rs.push_back(R);
    Ts.push_back(T);
    
    counter++;
  }
  
  while(counter < numImg) {
    Rs.push_back(R);
    Ts.push_back(T);

    counter++;
  }
  
  return true;
}

int App_main( int argc, char** argv ){
 
  std::string pathVideo = "/home/tgupta6/BundleAdjustment/dragon.mp4";
  std::string pathCameraExtrinsicsTxt = "/home/tgupta6/BundleAdjustment/Build/dragon_21_frames/camera_extrinsics.txt";
  std::string pathDepthDir = "/home/tgupta6/OpenDTAM/Cpp/Build/depthMaps";
  boost::filesystem::create_directory(pathDepthDir);
  
  Mat image, cameraMatrix, R, T;
  vector<Mat> images,Rs,ds,Ts;
  
  
  // Read Video Frames
  readVideo(pathVideo, images);
  int numImg = images.size();
  std::cout<< numImg << std::endl;
  // Read Camera Intrinsics
  cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  
  // Read Camera Extrinsics
  bool readCorrectly;
  
  readCorrectly = readCameraExtrinsics(numImg, 
				       pathCameraExtrinsicsTxt,
				       Rs,
				       Ts);
				       
  if(!readCorrectly) {
    std::cout << "Camera extrinsics were not read correctly" << std::endl;
    return 1;
  }
  
  
  //cv::namedWindow("Debug",1);
  //cv::imshow("Debug",images[0]);
  //cv::waitKey();
  Mat ret; // a place to return downloaded images to
  
  /*
  double reconstructionScale=1;
  int inc=1;
  int numImg;
  numImg = 50;
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
    double min, max;
    cv::minMaxLoc(image, &min, &max);
    std::cout << "Max: " << max << " Min: " << min << std::endl;
    tmp=Mat::zeros(image.rows,image.cols,CV_32FC3);
    randu(tmp,0,1);
    resize(image+tmp/255,image,Size(),reconstructionScale,reconstructionScale);
    images.push_back(image.clone());
    Rs.push_back(R.clone());
    Ts.push_back(T.clone());
    ds.push_back(d.clone());
       
  }
  */
  
  CudaMem cret(images[0].rows,images[0].cols,CV_32FC1);
  ret=cret.createMatHeader();
  
  int layers=64;
  int imagesPerCV=10;
  float occlusionThreshold=.05;
  Norm norm=L1T;
  /*
  cv::namedWindow("Debug",CV_WINDOW_NORMAL);
  cv::imshow("Debug", images[0]);
  cv::waitKey(10);
  */
  std::cout << images[11].channels() << std::endl;
  cv::imwrite(pathDepthDir + '/' + "0_th_img" + ".jpg", images[0]);
  for(int startAt=imagesPerCV; startAt < numImg-imagesPerCV; ++startAt) {
    std::cout<< "Attaching Cost Volume" << std::endl;
    //std::cout << startAt << std::endl;
    //std::cout << Rs[startAt] << Ts[startAt] << std::endl;
    
    CostVolume cv(images[startAt],(FrameID)startAt,layers,0.03,0.001,Rs[startAt],Ts[startAt],cameraMatrix,occlusionThreshold,norm);
    
    cv::gpu::Stream s;
    double totalscale=1.0;
    int tcount=0;
    int sincefail=0;
 
    for(int imageNum=startAt-imagesPerCV; imageNum<startAt; imageNum++) {
      //if(imageNum == startAt) { continue; }      
      std::cout<<"Updating Cost Volume using: "<< imageNum << std::endl;
      cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
      cudaDeviceSynchronize();
    }      
    
    //Attach optimizer
    std::cout << "Attaching Optimizer" << std::endl;
    Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(cv.baseImageGray,cv.cvStream);
    DepthmapDenoiseWeightedHuber& denoiser=*dp;
    Optimizer optimizer(cv);
    optimizer.initOptimization();
    GpuMat a(cv.loInd.size(),cv.loInd.type());
    cv.cvStream.enqueueCopy(cv.loInd,a);
    GpuMat d;
    denoiser.cacheGValues();
    ret=image*0;
      
    std::cout << "Optimizing Stage 1 (Using previous frames to compute current depth)";
    bool doneOptimizing; 
    do{
      a.download(ret);
      pfShow("A function", ret, 0, cv::Vec2d(0, layers));
      for (int i = 0; i < 10; i++) {
	d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
      }
      doneOptimizing=optimizer.optimizeA(d,a);
    }while(!doneOptimizing);
    std::cout << std::endl;

    // Attach tracker
    std::cout << "Attaching Tracker" << std::endl;
    Track tracker(cv);
    tracker.depth = optimizer.depthMap();
    for(int imageNum=startAt+1; imageNum < startAt + imagesPerCV; imageNum++) {
      std::cout << "Tracking and updating Cost Volume using: " << imageNum << std::endl;
      tracker.addFrame(images[imageNum]);
      tracker.align();
      LieToRT(tracker.pose, Rs[imageNum], Ts[imageNum]);
      cv.updateCost(images[imageNum], Rs[imageNum], Ts[imageNum]);
      cudaDeviceSynchronize();
    }
      
    // Optimize for the updated cost
    optimizer.initOptimization();
    cv.cvStream.enqueueCopy(cv.loInd,a);
    denoiser.cacheGValues();
    ret=image*0;

    std::cout << "Optimizing Stage 2 (Using next frames to compute current depth)";
    doneOptimizing = false; 
    do{
      a.download(ret);
      pfShow("A function", ret, 0, cv::Vec2d(0, layers));
      for (int i = 0; i < 10; i++) {
	d=denoiser(a,optimizer.epsilon,optimizer.getTheta());
      }
      doneOptimizing=optimizer.optimizeA(d,a);
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
    cv::imwrite(pathDepthDir + '/' + patch::to_string(startAt) + ".jpg", depthImg);

    // Final round of tracking
    /*
    std::cout << "Final round of tracking" << std::endl;
    tracker.depth = optimizer.depthMap();
    for(int imageNum=startAt - imagesPerCV; imageNum < startAt + imagesPerCV; imageNum++) {
      tracker.addFrame(images[imageNum]);
      tracker.align();
      LieToRT(tracker.pose, Rs[imageNum], Ts[imageNum]);
    }
    */
  }
    
  Stream::Null().waitForCompletion();
  return 0;
}


