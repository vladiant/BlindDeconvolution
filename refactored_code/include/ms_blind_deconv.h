/*
 * ms_blind_deconv.h
 *
 *  Created on: Jan 9, 2014
 *      Author: vladiant
 */

#ifndef MS_BLIND_DECONV_H_
#define MS_BLIND_DECONV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "deconv_opts.h"

//
// Do multi-scale blind deconvolution given input file name and options
// structure opts. Returns a double deblurred image along with estimated
// kernel. Following the kernel estimation, a non-blind deconvolution is run.
//
// Copyright (2011): Dilip Krishnan, Rob Fergus, New York University.
//

void ms_blind_deconv(cv::Mat& blurredImage, const BlindDeblurOptions& opts,
		cv::Mat& kernelImage, cv::Mat& deblurredImage, BlindDeblurContext& aContext);

void init_kernel(int minsize, cv::Mat& k);

void center_kernel_separate(const cv::Mat& x, const cv::Mat& y,
		const cv::Mat& k, cv::Mat& x_shifted, cv::Mat& y_shifted,
		cv::Mat& k_shifted);

void cgDeblur(const cv::Mat& blurredImage, const cv::Mat& kernel,
		cv::Mat& deblurredImage);

void rlDeblur(const cv::Mat& blurredImage, const cv::Mat& kernel,
		cv::Mat& deblurredImage);

#endif /* MS_BLIND_DECONV_H_ */
