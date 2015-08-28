#include <fstream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "LKTrackBA.hpp"


// Parameters for goodFeaturesToTrack function in OpenCV
const int MAX_CORNERS = 1000;
const double MAX_LEVEL = 0.1;
const double MIN_DIST = 10;
const int BLOCK_SIZE = 5;
const bool USE_HARRIS_CORNERS = false;
const double K = 0.04;

// Parameters for cornerSubPix
const cv::Size WIN_SIZE_SUBPIX(5,5);
const cv::Size ZERO_ZONE(-1,1);
const cv::TermCriteria CRITERIA_SUBPIX(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);

// Threshold for adding new detected points to existing tracked points
const double DIST_THRESH = 20;

// Parameters for LKT
const cv::Size WIN_SIZE(15,15);
const int MAX_PYR_LEVEL = 3;
const cv::TermCriteria CRITERIA(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.001);
const int OP_FLAG = cv::OPTFLOW_USE_INITIAL_FLOW;
const double MIN_EIG_THRESH = 0.01;


LKTrackerAndBundleAdjuster::LKTrackerAndBundleAdjuster(const cv::Mat &initialFrame) {
  prevFrame = initialFrame.clone();
  lastTrackNum = -1;
  lastImgNum = 0;
}

void LKTrackerAndBundleAdjuster::addFrame(const cv::Mat &frame){
  std::vector<cv::Point2f> detectedPts;
  cv::goodFeaturesToTrack(prevFrame, 
			  detectedPts, 
			  MAX_CORNERS, 
			  MAX_LEVEL, 
			  MIN_DIST, 
			  cv::Mat(), 
			  BLOCK_SIZE,
			  USE_HARRIS_CORNERS,
			  K);
  cv::cornerSubPix(prevFrame, detectedPts, WIN_SIZE_SUBPIX, ZERO_ZONE, CRITERIA_SUBPIX);
  
  //   Check the distance of the detected point to all tracked points on the prevFrame 
  //   and add the point if found to be sufficiently far away from all of them
  cv::Point2f diff;
  double dist, minDist;
  bool doNotAddPt;
  
  for(int i=0; i<detectedPts.size(); ++i) {
    doNotAddPt = false;
    minDist = frame.rows + frame.cols;
    for(int j=0; j < prevFramePts.size() && !doNotAddPt; ++j) {
      diff = detectedPts.at(i) - prevFramePts.at(j);
      dist = cv::norm(diff);
      if(dist < minDist) { minDist = dist; }
      if(minDist < DIST_THRESH) { doNotAddPt = true; }
    }

    if(!doNotAddPt) {
      // add point to the list of points to be tracked in the current frame
      prevFramePts.push_back(detectedPts.at(i));

      // add the point to the list of all tracked 2d image points 
      lastTrackNum++;
      prevFrameTrackNums.push_back(lastTrackNum);
      tracked2dPoint new2dPoint;
      new2dPoint.imgPt = detectedPts.at(i);
      new2dPoint.trackNum = lastTrackNum;
      new2dPoint.imgNum = lastImgNum;
      allTrackedPts.push_back(new2dPoint);

      // since these are new tracks also create and add corresponding 3d points
      tracked3dPoint new3dPoint;
      new3dPoint.num2dPts = 1;
      trackNum2WorldCoord[lastTrackNum] = new3dPoint;
    }
  }

  // Track the points using Lucas Kanade
  std::cout << "NumPts: " << prevFramePts.size() << std::endl;
  std::vector<cv::Point2f> nextFramePts = prevFramePts;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prevFrame, 
			   frame, 
			   prevFramePts, 
			   nextFramePts, 
			   status, 
			   err,
			   WIN_SIZE,
			   MAX_PYR_LEVEL,
			   CRITERIA,
			   OP_FLAG,
			   MIN_EIG_THRESH);
  
  // Clear prevFramePts and add the nextFramePt if successfully tracked and 
  prevFramePts.clear();
  std::vector<int> nextFrameTrackNums;
  lastImgNum++;
  for(int i=0; i < nextFramePts.size(); i++) {
    if(status.at(i)==1) {
      // add point to the list of points to be tracked in the current frame
      prevFramePts.push_back(nextFramePts.at(i));

      // add the point to the list of all tracked 2d image points 
      tracked2dPoint new2dPoint;
      new2dPoint.imgPt = nextFramePts.at(i);
      new2dPoint.trackNum = prevFrameTrackNums.at(i);
      new2dPoint.imgNum = lastImgNum;
      allTrackedPts.push_back(new2dPoint);

      // Update track information
      trackNum2WorldCoord[prevFrameTrackNums.at(i)].num2dPts++;
      nextFrameTrackNums.push_back(prevFrameTrackNums.at(i));
    }
  }
  
  // Update prevFrameTrackNums
  prevFrameTrackNums = nextFrameTrackNums;
  prevFrame = frame.clone();
}

void LKTrackerAndBundleAdjuster::visualizeTracking() {
  // read points into Keypoint datastructure
  std::vector<cv::KeyPoint> trackedKpts;
  const float KPT_SIZE = 1;
  for(int i=0; i<prevFramePts.size(); ++i) {
    trackedKpts.push_back(cv::KeyPoint(prevFramePts.at(i),KPT_SIZE));
  }
  
  // draw keypoints on the image
  cv::drawKeypoints(prevFrame, trackedKpts, imgWithKpts, cv::Scalar(0,1,0));
  
  // displayKeypoints
  cv::namedWindow("Image with Keypoints", CV_WINDOW_NORMAL);
  cv::imshow("Image with Keypoints", imgWithKpts);
  cv::waitKey(1);
}


//***********Bundle Adjustment Part**********//

struct Reprojection_Error {
  double observed_x, observed_y;
  Reprojection_Error(double observed_x_, double observed_y_) {
    observed_x = observed_x_;
    observed_y = observed_y_;
  }

  template <typename T>
  bool operator()(const T* const camera_extrinsic, const T* point, T* residual) const {
    // Rotate point
    T camera_coord[3];
    ceres::AngleAxisRotatePoint(camera_extrinsic, point, camera_coord);

    // Translate rotated point
    camera_coord[0] += camera_extrinsic[3]; camera_coord[1] += camera_extrinsic[4]; camera_coord[2] += camera_extrinsic[5];

    // Apply camera intrinsic matrix
    T focal_length_x = (T)fx;
    T focal_length_y = (T)fy;
    T px = (T)cx;
    T py = (T)cy;

    T projected_pt[2];
    projected_pt[0] = focal_length_x*camera_coord[0]/camera_coord[2] + px;
    projected_pt[1] = focal_length_y*camera_coord[1]/camera_coord[2] + py;

    residual[0] = T(observed_x) - projected_pt[0];
    residual[1] = T(observed_y) - projected_pt[1];

    return true;
  }
};

void LKTrackerAndBundleAdjuster::solveWithCeres(const std::string &outputDir) {
  // Initialize the variables to solve for
  //  double camera_intrinsics[3] = {1000, (double)prevFrame.cols, (double)prevFrame.rows};
  double** camera_extrinsics = new double* [lastImgNum+1];
  for(int i=0; i < lastImgNum+1; i++) {
    camera_extrinsics[i] = new double [6];
    for(int j=0; j < 6; j++) {
      camera_extrinsics[i][j] = 0.1;
    }
  }
  
  for(int i=0; i < trackNum2WorldCoord.size(); i++) {
    for(int j=0; j < 3; j++) {
      trackNum2WorldCoord[i].pt[j] = 0; 
    }
  }
  
  // Build model
  ceres::Problem problem;
  
  // Setup a cost function, one for each tracked2dPoint
  for(int i=0; i < allTrackedPts.size(); i++) {
    if(trackNum2WorldCoord[allTrackedPts.at(i).trackNum].num2dPts < 2) {
      continue;
    }

    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<Reprojection_Error,2,6,3>(new Reprojection_Error(allTrackedPts.at(i).imgPt.x, allTrackedPts.at(i).imgPt.y));
    problem.AddResidualBlock(cost_function, 
			     new ceres::HuberLoss(2), 
			     camera_extrinsics[allTrackedPts.at(i).imgNum],
			     trackNum2WorldCoord[allTrackedPts.at(i).trackNum].pt);
  }

  ceres::Solver::Options options;
  options.use_explicit_schur_complement = true;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  // Write the World coordinates to a text file
  std::ofstream world_coord_txt;
  std::string txtFileName = outputDir + "/World_Coordinates.txt";
  world_coord_txt.open(txtFileName);
  for(int i=0; i<lastTrackNum+1; i++) {
    if(trackNum2WorldCoord[i].num2dPts < 2) {
      continue;
    }
    world_coord_txt << trackNum2WorldCoord[i].pt[0] << " " << trackNum2WorldCoord[i].pt[1] << " " << trackNum2WorldCoord[i].pt[2] << std::endl;
  }
  world_coord_txt.close();

  // Write the Camera Extrinsics to a text file
  std::ofstream camera_extrinsics_txt;
  txtFileName = outputDir + "/camera_extrinsics.txt";
  camera_extrinsics_txt.open(txtFileName);
  camera_extrinsics_txt << lastImgNum+1 << std::endl;
  double* R = new double [9];
  for(int i=0; i < lastImgNum+1; i++) {
    // Write frame number
    camera_extrinsics_txt << i << " ";

    // Write camera translation
    for(int j=3; j < 6; ++j) {
      camera_extrinsics_txt << camera_extrinsics[i][j] << " ";
    }

    // Convert angle axis representation to rotation matrix (column major)
    ceres::AngleAxisToRotationMatrix<double>(camera_extrinsics[i], R);

    // Write camera rotation
    for(int j=0; j < 8; ++j) {
      camera_extrinsics_txt << R[j] << " ";
    }
    camera_extrinsics_txt << R[8] << std::endl;
  }
  camera_extrinsics_txt.close();
}
