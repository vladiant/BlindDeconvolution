/*
 * fast_deconv_bregman.cpp
 *
 *  Created on: 15.02.2015
 *      Author: vladiant
 */

#include "fast_deconv_bregman.h"

#include <iostream>
#include <vector>

#include "pcg_kernel_irls_conv.h"
#include "solve_image_bregman.h"

void fast_deconv_bregman(const cv::Mat& f, const cv::Mat& k, float lambda,
                         float alpha, cv::Mat& g) {
  //
  // fast solver for the non-blind deconvolution problem: min_g \lambda/2 |g
  // \oplus k
  // - f|^2. We use a splitting trick as
  // follows: introduce a (vector) variable w, and rewrite the original
  // problem as: min_ {g,w,b}\lambda / 2 | g
  // \oplus k
  // - g|^2 + \beta/2 |w -
  // \nabla g - b|^2, and then we use alternations on g, w
  // and b to update each one in turn. b is the Bregman variable. beta is
  // fixed. An alternative is to use continuation but then we need to set a
  // beta regime. Based on the NIPS 2009 paper of Krishnan and Fergus "Fast
  // Image Deconvolution using Hyper-Laplacian Priors"
  //

  float beta = 10;

  int initer_max = 1;
  int outiter_max = 20;

  int m = f.cols;
  int n = f.rows;

  // initialize
  f.copyTo(g);

  // make sure k is odd-sized
  if (k.rows % 2 != 1 || (k.cols % 2 != 1)) {
    std::cout << "Error - blur kernel k must be odd-sized.\n" << std::endl;
    return;
  }
  int ks = floor((k.rows - 1) / 2);

  cv::Mat dx(1, 2, CV_32FC1);
  dx.at<float>(cv::Point(0, 0)) = 1.0;
  dx.at<float>(cv::Point(1, 0)) = -1.0;

  cv::Mat dy(2, 1, CV_32FC1);
  dy.at<float>(cv::Point(0, 0)) = 1.0;
  dy.at<float>(cv::Point(0, 1)) = -1.0;

  cv::Mat dxt(1, 2, CV_32FC1);
  dxt.at<float>(cv::Point(0, 0)) = -1.0;
  dxt.at<float>(cv::Point(1, 0)) = 1.0;

  cv::Mat dyt(2, 1, CV_32FC1);
  dyt.at<float>(cv::Point(0, 0)) = -1.0;
  dyt.at<float>(cv::Point(0, 1)) = 1.0;

  cv::Mat Ktf, KtK, DtD, Fdx, Fdy;
  computeConstants(f, k, dx, dy, Ktf, KtK, DtD, Fdx, Fdy);

  // TODO: Debug print
  //	cv::namedWindow("Ktf", cv::WINDOW_NORMAL);
  //	cv::imshow("Ktf", Ktf);
  //	cv::namedWindow("KtK", cv::WINDOW_NORMAL);
  //	cv::imshow("KtK", KtK);
  //	cv::namedWindow("DtD", cv::WINDOW_NORMAL);
  //	cv::imshow("DtD", DtD);
  //	cv::namedWindow("Fdx", cv::WINDOW_NORMAL);
  //	cv::imshow("Fdx", Fdx);
  //	cv::namedWindow("Fdy", cv::WINDOW_NORMAL);
  //	cv::imshow("Fdy", Fdy);
  //	cv::waitKey(0);

  cv::Mat gx;
  cv::filter2D(g, gx, -1, dx);

  cv::Mat gy;
  cv::filter2D(g, gy, -1, dy);

  cv::Mat fx;
  cv::filter2D(f, fx, -1, dx);

  cv::Mat fy;
  cv::filter2D(f, fy, -1, dy);

  // TODO: Debug print
  //	cv::namedWindow("gx", cv::WINDOW_NORMAL);
  //	cv::imshow("gx", gx);
  //	cv::namedWindow("gy", cv::WINDOW_NORMAL);
  //	cv::imshow("gy", gy);
  //	cv::namedWindow("fx", cv::WINDOW_NORMAL);
  //	cv::imshow("fx", fx);
  //	cv::namedWindow("fy", cv::WINDOW_NORMAL);
  //	cv::imshow("fy", fy);
  //	cv::waitKey(0);

  ks = k.rows;
  int ks2 = floor(ks / 2);

  // store some of the statistics
  std::vector<float> lcost;
  std::vector<float> pcost;
  int outiter = 0;

  cv::Mat bx = gx.clone();
  bx = cv::Scalar(0);
  cv::Mat by = gy.clone();
  by = cv::Scalar(0);
  cv::Mat wx = gx.clone();
  cv::Mat wy = gy.clone();

  int totiter = 0;
  cv::Mat gk;
  cv::filter2D(g, gk, -1, k);

  // TODO: Debug print
  //	cv::namedWindow("gk", cv::WINDOW_NORMAL);
  //	cv::imshow("gk", gk);
  //	cv::waitKey(0);

  if (lcost.size() > totiter) {
    lcost[totiter] = (lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                     cv::norm(gk - f, cv::NORM_L2);
    cv::Mat gxPow;
    cv::pow(cv::abs(gx), alpha, gxPow);
    cv::Mat gyPow;
    cv::pow(cv::abs(gy), alpha, gyPow);
    pcost[totiter] = cv::sum(gxPow)[0] + cv::sum(gyPow)[0];

  } else {
    lcost.push_back((lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                    cv::norm(gk - f, cv::NORM_L2));
    cv::Mat gxPow;
    cv::pow(cv::abs(gx), alpha, gxPow);
    cv::Mat gyPow;
    cv::pow(cv::abs(gy), alpha, gyPow);
    pcost.push_back((lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                    cv::norm(gk - f, cv::NORM_L2));
  }

  for (int outiter = 0; outiter < outiter_max; outiter++) {
    std::cout << "Outer iteration " << outiter << std::endl;
    int initer = 0;

    for (int initer = 0; initer < initer_max; initer++) {
      totiter = totiter + 1;

      if (alpha == 1) {
        cv::Mat tmpx = beta * (gx + bx);
        float betax = beta;
        tmpx = tmpx / betax;

        cv::Mat tmpy = beta * (gy + by);
        float betay = beta;
        tmpy = tmpy / betay;

        cv::max(cv::abs(tmpx) - 1.0 / betax, 0, wx);
        cv::max(cv::abs(tmpy) - 1.0 / betay, 0, wy);

        cv::multiply(wx, cv::abs(tmpx), wx);
        cv::divide(wx, tmpx, wx);
        cv::patchNaNs(wx, 0);

        cv::multiply(wy, cv::abs(tmpy), wy);
        cv::divide(wy, tmpy, wy);
        cv::patchNaNs(wy, 0);

      } else {
        BregmanImageSolver solver;
        solver(gx + bx, beta, alpha, wx);
        solver(gy + by, beta, alpha, wy);
      }

      // TODO: Debug print
      //			cv::namedWindow("wx", cv::WINDOW_NORMAL);
      //			cv::imshow("wx", wx);
      //			cv::namedWindow("wy", cv::WINDOW_NORMAL);
      //			cv::imshow("wy", wy);
      //			cv::waitKey(0);

      bx = bx - wx + gx;
      by = by - wy + gy;

      cv::Mat wx1, wy1;
      cv::filter2D(wx - bx, wx1, -1, dxt);
      cv::filter2D(wy - by, wy1, -1, dyt);

      // TODO: Debug print
      //			cv::namedWindow("wx1", cv::WINDOW_NORMAL);
      //			cv::imshow("wx1", wx1);
      //			cv::namedWindow("wy1", cv::WINDOW_NORMAL);
      //			cv::imshow("wy1", wy1);
      //			cv::waitKey(0);

      cv::Mat gprev = g.clone();
      cv::Mat gxprev = gx.clone();
      cv::Mat gyprev = gy.clone();

      cv::Mat wxy1 = wx1 + wy1;

      std::vector<cv::Mat> wxy1s;
      cv::split(wxy1, wxy1s);
      for (size_t i = 0; i < wxy1s.size(); i++) {
        cv::dft(wxy1s[i], wxy1s[i]);
      }
      cv::merge(wxy1s, wxy1);

      cv::Mat num = lambda * Ktf + beta * wxy1;
      cv::Mat denom = lambda * KtK + beta * DtD;

      // TODO: Debug print
      //			cv::namedWindow("num", cv::WINDOW_NORMAL);
      //			cv::imshow("num", num);
      //			cv::namedWindow("denom", cv::WINDOW_NORMAL);
      //			cv::imshow("denom", denom);
      //			cv::waitKey(0);

      cv::Mat Fg = g.clone();
      std::vector<cv::Mat> Fgs;
      cv::split(Fg, Fgs);
      std::vector<cv::Mat> nums;
      cv::split(num, nums);
      invertFftMatrix(denom);
      for (size_t i = 0; i < wxy1s.size(); i++) {
        cv::mulSpectrums(nums[i], denom, Fgs[i], cv::DFT_COMPLEX_OUTPUT);
        cv::dft(Fgs[i], Fgs[i],
                cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
      }
      cv::merge(Fgs, Fg);

      Fg.copyTo(g);

      // TODO: Debug print
      //			cv::namedWindow("g", cv::WINDOW_NORMAL);
      //			cv::imshow("g", g);
      //			cv::waitKey(0);

      cv::filter2D(g, gx, -1, dx);
      cv::filter2D(g, gy, -1, dy);
      cv::filter2D(g, gk, -1, k);

      if (lcost.size() > totiter) {
        lcost[totiter] = (lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                         cv::norm(gk - f, cv::NORM_L2);
        cv::Mat gxPow;
        cv::pow(cv::abs(gx), alpha, gxPow);
        cv::Mat gyPow;
        cv::pow(cv::abs(gy), alpha, gyPow);
        pcost[totiter] = cv::sum(gxPow)[0] + cv::sum(gyPow)[0];

      } else {
        lcost.push_back((lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                        cv::norm(gk - f, cv::NORM_L2));
        cv::Mat gxPow;
        cv::pow(cv::abs(gx), alpha, gxPow);
        cv::Mat gyPow;
        cv::pow(cv::abs(gy), alpha, gyPow);
        pcost.push_back((lambda / 2.0) * cv::norm(gk - f, cv::NORM_L2) *
                        cv::norm(gk - f, cv::NORM_L2));
      }
    }
  }
}

void computeConstants(const cv::Mat& f, const cv::Mat& k, const cv::Mat& dx,
                      const cv::Mat& dy, cv::Mat& Ktf, cv::Mat& KtK,
                      cv::Mat& DtD, cv::Mat& Fdx, cv::Mat& Fdy) {
  int sizefX = f.cols;
  int sizefY = f.rows;

  cv::Mat otfk(sizefY, sizefX, CV_32FC1);
  otfk = cv::Scalar(0);
  copyKernelToImage((float*)k.data, k.cols, k.rows, k.cols, (float*)otfk.data,
                    otfk.cols, otfk.rows, otfk.cols);
  cv::dft(otfk, otfk);

  cv::Mat fFFT(sizefY, sizefX, CV_32FC1);

  std::vector<cv::Mat> fChannels;
  cv::split(f, fChannels);
  std::vector<cv::Mat> KtfChannels(fChannels.size());

  for (size_t i = 0; i < fChannels.size(); i++) {
    cv::dft(fChannels[i], fChannels[i]);
    cv::mulSpectrums(fChannels[i], otfk, KtfChannels[i], cv::DFT_COMPLEX_OUTPUT,
                     true);
  }

  cv::merge(fChannels, fFFT);
  cv::merge(KtfChannels, Ktf);

  cv::mulSpectrums(otfk, otfk, KtK, cv::DFT_COMPLEX_OUTPUT, true);

  cv::Mat dFFT(sizefY, sizefX, CV_32FC1);

  std::vector<cv::Mat> dxChannels, dyChannels;
  cv::split(dx, dxChannels);
  cv::split(dy, dyChannels);

  dFFT = cv::Scalar(0);
  copyKernelToImage((float*)dx.data, dx.cols, dx.rows, dx.cols,
                    (float*)dFFT.data, dFFT.cols, dFFT.rows, dFFT.cols);
  cv::dft(dFFT, dFFT);
  cv::mulSpectrums(dFFT, dFFT, Fdx, cv::DFT_COMPLEX_OUTPUT, true);

  dFFT = cv::Scalar(0);
  copyKernelToImage((float*)dy.data, dy.cols, dy.rows, dy.cols,
                    (float*)dFFT.data, dFFT.cols, dFFT.rows, dFFT.cols);
  cv::dft(dFFT, dFFT);
  cv::mulSpectrums(dFFT, dFFT, Fdy, cv::DFT_COMPLEX_OUTPUT, true);

  cv::add(Fdx, Fdy, DtD);
}

void invertFftMatrix(cv::Mat& matrixFft) {
  float realValue, imaginaryValue,
      sum;  // temporal variables for matrix inversion

  uchar* matrixData = matrixFft.data;

  auto matrixStep = matrixFft.step;

  int matrixWidth = matrixFft.cols;
  int matrixHeight = matrixFft.rows;

  int matrixHalfWidth =
      ((matrixWidth % 2 == 0) ? matrixWidth - 2 : matrixWidth - 1);
  int matrixHalfHeight =
      ((matrixHeight % 2 == 0) ? matrixHeight - 2 : matrixHeight - 1);

  // sets upper left
  float upperLeftValue = ((float*)matrixData)[0];
  if (upperLeftValue != 0) {
    ((float*)matrixData)[0] = 1.0 / upperLeftValue;
  } else {
    ((float*)matrixData)[0] = 0.0;
  }

  // set first column
  for (int row = 1; row < matrixHalfHeight; row += 2) {
    realValue = ((float*)(matrixData + row * matrixStep))[0];
    imaginaryValue = ((float*)(matrixData + (row + 1) * matrixStep))[0];
    sum = realValue * realValue + imaginaryValue * imaginaryValue;
    if (sum != 0) {
      ((float*)(matrixData + row * matrixStep))[0] = realValue / sum;
      ((float*)(matrixData + (row + 1) * matrixStep))[0] =
          -imaginaryValue / sum;
    } else {
      ((float*)(matrixData + row * matrixStep))[0] = 0.0;
      ((float*)(matrixData + (row + 1) * matrixStep))[0] = 0.0;
    }
  }

  // sets down left if needed
  if (matrixHeight % 2 == 0) {
    float downLeftValue =
        ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0];
    if (downLeftValue != 0) {
      ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] =
          1.0 / downLeftValue;
    } else {
      ((float*)(matrixData + (matrixHeight - 1) * matrixStep))[0] = 0.0;
    }
  }

  if (matrixWidth % 2 == 0) {
    // sets upper right
    float upperLeftValue = ((float*)matrixData)[matrixWidth - 1];
    if (upperLeftValue != 0) {
      ((float*)matrixData)[matrixWidth - 1] = 1.0 / upperLeftValue;
    } else {
      ((float*)matrixData)[matrixWidth - 1] = 0.0;
    }

    // set last column
    for (int row = 1; row < matrixHalfHeight; row += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[matrixWidth - 1];
      imaginaryValue =
          ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue;
      if (sum != 0) {
        ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] =
            realValue / sum;
        ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] =
            -imaginaryValue / sum;
      } else {
        ((float*)(matrixData + row * matrixStep))[matrixWidth - 1] = 0.0;
        ((float*)(matrixData + (row + 1) * matrixStep))[matrixWidth - 1] = 0.0;
      }
    }

    // sets down right
    if (matrixHeight % 2 == 0) {
      float downRightValue =
          ((float*)(matrixData +
                    (matrixHeight - 1) * matrixStep))[matrixWidth - 1];
      if (downRightValue != 0) {
        ((float*)(matrixData +
                  (matrixHeight - 1) * matrixStep))[matrixWidth - 1] =
            1.0 / downRightValue;
      } else {
        ((float*)(matrixData +
                  (matrixHeight - 1) * matrixStep))[matrixWidth - 1] = 0.0;
      }
    }
  }

  for (int row = 0; row < matrixHeight; row++) {
    for (int col = 1; col < matrixHalfWidth; col += 2) {
      realValue = ((float*)(matrixData + row * matrixStep))[col];
      imaginaryValue = ((float*)(matrixData + row * matrixStep))[col + 1];
      sum = realValue * realValue + imaginaryValue * imaginaryValue;
      if (sum != 0) {
        ((float*)(matrixData + row * matrixStep))[col] = realValue / sum;
        ((float*)(matrixData + row * matrixStep))[col + 1] =
            (-imaginaryValue / sum);
      } else {
        ((float*)(matrixData + row * matrixStep))[col] = 0.0;
        ((float*)(matrixData + row * matrixStep))[col + 1] = 0.0;
      }
    }
  }
}
