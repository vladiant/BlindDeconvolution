#pragma once

#include <opencv2/core/core.hpp>

class Convolver {
 public:
  void operator()(const float* imageData, int imageWidth, int imageHeight,
                  int imagePpln, const float* kernelData, int kernelWidth,
                  int kernelHeight, int kernelPpln, float* convolutionData,
                  int convolutionWidth, int convolutionHeight,
                  int convolutionPpln);

 private:
  // TODO: Pre-allocation of image buffers - size of the input image
  cv::Mat imageFFT;
  cv::Mat kernelFFT;
};
