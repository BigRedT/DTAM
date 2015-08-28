#ifndef _LKTRACKBA_H_
#define _LKTRACKBA_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>

extern const double fx, fy, cx, cy;

struct tracked2dPoint {
  cv::Point2f imgPt;
  int trackNum;
  int imgNum;
};

struct tracked3dPoint {
  int num2dPts;
  double pt[3];
};

class LKTrackerAndBundleAdjuster {
public:
  cv::Mat prevFrame;
  cv::Mat imgWithKpts;
  std::vector<tracked2dPoint> allTrackedPts;
  std::unordered_map<int,tracked3dPoint> trackNum2WorldCoord;
  std::vector<cv::Point2f> prevFramePts;
  std::vector<int> prevFrameTrackNums;
  int lastTrackNum;
  int lastImgNum;

public:
  LKTrackerAndBundleAdjuster(const cv::Mat &initialFrame); 
  void addFrame(const cv::Mat &frame);
  void visualizeTracking();
  void solveWithCeres(const std::string &outputDir);
};

#endif 
