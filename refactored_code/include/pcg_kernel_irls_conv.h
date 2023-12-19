/*
 * pcg_kernel_irls_conv.h
 *
 *  Created on: Jan 19, 2014
 *      Author: vladiant
 */

#ifndef PCG_KERNEL_IRLS_CONV_H_
#define PCG_KERNEL_IRLS_CONV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "deconv_opts.h"

void pcg_kernel_irls_conv(const cv::Mat& k_init, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>& Y, const BlindDeblurOptions& opts, const BlindDeblurContext& aContext, 
		cv::Mat& k_out);

void local_cg(const cv::Mat& k, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>& flipX, int ks, const cv::Mat& weights_l1,
		const cv::Mat& rhs, float tol, int max_its, cv::Mat& k_out);

void pcg_kernel_core_irls_conv(const cv::Mat& k, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>&flipX, int ks, const cv::Mat& weights_l1,
		cv::Mat& out);

void copyImageToKernel(const float* pImagedata, int imageCols, int imageRows,
		int imagePpln, float* pKerneldata, int kernelCols, int kernelRows,
		int kernelPpln);

void copyKernelToImage(const float* pKerneldata, int kernelCols, int kernelRows,
		int kernelPpln, float* pImagedata, int imageCols, int imageRows,
		int imagePpln);

#endif /* PCG_KERNEL_IRLS_CONV_H_ */
