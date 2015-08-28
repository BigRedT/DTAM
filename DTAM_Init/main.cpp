#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "LKTrackBA.hpp"

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
const double fx = 1745.75;
const double fy = 1760.71;
const double cx = 997.65;
const double cy = 558.68;
const double k1 = 0.0660;
const double k2 = 0.2850;
const double k3 = -1.6434;
const double p1 = -0.0008;
const double p2 = -0.0015;

// Number of frames to run on
const int NUM_FRAMES = 20;
const int FRAME_WIDTH = 1920;
const int FRAME_HEIGHT = 1088;

void undistortFrame(const cv::Mat &frame, cv::Mat &undistortedFrame) {
  cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
  cv::Mat distCoeffs = (cv::Mat_<double>(5,1) << k1, k2, p1, p2, k3);
  cv::undistort(frame, undistortedFrame, cameraMatrix, distCoeffs);
}

int main() {
  std::string filename = "/home/tgupta6/BundleAdjustment/jenga.mp4";
  std::string outputDir = "/home/tgupta6/BundleAdjustment/Build/jenga_21_frames";
  std::string trackedFrameName;
  std::ostringstream convert;

  // open video file to be read
  cv::VideoCapture videoObj;
  videoObj.open(filename);
  if(!videoObj.isOpened()) {
    std::cout << "The file could not be openned" << std::endl;
    return 1;
  }
  else {
    std::cout << "The file was successfully opened" << std::endl;
  }

  cv::namedWindow("Video frames", CV_WINDOW_NORMAL);
  
  cv::Mat _frame, __frame, __frame_resized, frame;
  videoObj >> _frame;
  if(_frame.empty()) {
    std::cout << "No frames to process" << std::endl;
    return 1;
  }
  cv::cvtColor(_frame, __frame, CV_BGR2GRAY);
  cv::resize(__frame, __frame_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
  undistortFrame(__frame_resized, frame);

  trackedFrameName = outputDir + "/tracked_0.jpg";
  cv::imwrite(trackedFrameName, frame);

  LKTrackerAndBundleAdjuster tracker(frame);

  int count = 0;
  while(count<NUM_FRAMES) {
    count++;

    convert.str("");
    convert.clear();
    convert << count;

    videoObj >> _frame;
    if(_frame.empty()) {
      std::cout << "Processed entire video" << std::endl;
      return 0;
    }
    cv::cvtColor(_frame, __frame, CV_BGR2GRAY);
    cv::resize(__frame, __frame_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
    undistortFrame(__frame_resized, frame);
    
    tracker.addFrame(frame);
    std::cout << "lastTrackNum: " << tracker.lastTrackNum << std::endl;
    std::cout << "lastImgNum: " << tracker.lastImgNum << std::endl;

    tracker.visualizeTracking();
    
    trackedFrameName = outputDir + "/tracked_" + convert.str() + ".jpg";
    cv::imwrite(trackedFrameName, tracker.imgWithKpts);
    
  }
  tracker.solveWithCeres(outputDir);
  
  return 0;
}
