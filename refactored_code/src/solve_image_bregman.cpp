#include "solve_image_bregman.h"

#include <cmath>

double BregmanImageSolver::weightFunction(double w, double alpha, double beta,
                                          double v) {
  return v - alpha * pow(fabs(w), alpha - 2.0) * w / beta;
}

double BregmanImageSolver::solveWeightFunction(double startX, double alpha,
                                               double beta, double v) {
  const double EPSILON = 1e-6;
  const int MAX_ITERATIONS = 100;

  double oldSolution = startX;
  double solution = oldSolution;

  double currentDiff = 0;

  int iteration = 0;
  do {
    oldSolution = solution;
    solution = weightFunction(solution, alpha, beta, v);
    iteration++;

    currentDiff = fabs(solution - oldSolution);

  } while (currentDiff > EPSILON && iteration < MAX_ITERATIONS);

  if (std::isnan(solution)) {
    solution = 0.0;
  }

  if (iteration >= MAX_ITERATIONS) {
    solution = 0;
  }

  return solution;
}

void BregmanImageSolver::prepareWeightLut(float beta, float alpha) {
  static float calculatedBeta = 0;

  if (beta != calculatedBeta) {
    calculatedBeta = beta;
  } else {
    return;
  }

  for (int i = 0; i < WEIGHT_LUT_SIZE; i++) {
    float value = 32.0 * (i - WEIGHT_LUT_SIZE / 2.0) / WEIGHT_LUT_SIZE;

    float x = solveWeightFunction(value, alpha, beta, value);

    weightLut[i] = x;
  }
}

float BregmanImageSolver::WeightCalcAlpha(float v, float beta) {
  int index = (v + 16) * WEIGHT_LUT_SIZE / 32;

  if (index < 0) {
    return weightLut[0];

  } else if (index >= WEIGHT_LUT_SIZE - 1) {
    return weightLut[WEIGHT_LUT_SIZE - 1];

  } else {
    float indexValue = 32.0 * (index - WEIGHT_LUT_SIZE / 2.0) / WEIGHT_LUT_SIZE;
    float interpolationWeight = v - indexValue;

    float value =
        weightLut[index] +
        interpolationWeight * (weightLut[index + 1] - weightLut[index]);
    return value;
  }
}

void BregmanImageSolver::operator()(const cv::Mat& image, float beta,
                                    float alpha, cv::Mat& wx) {
  prepareWeightLut(beta, alpha);

  int imageHeight = image.rows;
  int imageWidth = image.cols;
  int imageChannels = image.channels();
  float* inputImageData = (float*)image.data;
  float* outputImageData = (float*)wx.data;

  for (int channel = 0; channel < imageChannels; channel++) {
    for (int row = 0; row < imageHeight; row++) {
      for (int col = 0; col < imageWidth; col++) {
        float value =
            inputImageData[imageChannels * (col + row * imageWidth) + channel];
        value = WeightCalcAlpha(value, beta);
        outputImageData[imageChannels * (col + row * imageWidth) + channel] =
            value;
      }
    }
  }
}
