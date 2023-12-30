#pragma once

#include <opencv2/core/core.hpp>
#include <vector>

class BregmanImageSolver {
 public:
  void operator()(const cv::Mat& image, float beta, float alpha, cv::Mat& wx);

 private:
  float WeightCalcAlpha(float v, float beta);
  void prepareWeightLut(float beta, float alpha);
  double solveWeightFunction(double startX, double alpha, double beta,
                             double v);
  double weightFunction(double w, double alpha, double beta, double v);

  static constexpr int WEIGHT_LUT_SIZE = 4096;
  std::vector<float> weightLut{WEIGHT_LUT_SIZE};
};
