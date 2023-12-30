/*
 * fast_deconv_bregman.h
 *
 *  Created on: 15.02.2015
 *      Author: vladiant
 */

#ifndef FAST_DECONV_BREGMAN_H_
#define FAST_DECONV_BREGMAN_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void fast_deconv_bregman(const cv::Mat& f, const cv::Mat& k, float lambda,
                         float alpha, cv::Mat& g);

void computeConstants(const cv::Mat& f, const cv::Mat& k, const cv::Mat& dx,
                      const cv::Mat& dy, cv::Mat& Ktf, cv::Mat& KtK,
                      cv::Mat& DtD, cv::Mat& Fdx, cv::Mat& Fdy);

void invertFftMatrix(cv::Mat& matrixFft);

#endif /* FAST_DECONV_BREGMAN_H_ */
