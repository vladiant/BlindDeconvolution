/*
 * ms_blind_deconv.cpp
 *
 *  Created on: Jan 9, 2014
 *      Author: vladiant
 */

#include "ms_blind_deconv.h"
#include "ss_blind_deconv.h"
#include "fast_deconv_bregman.h"

#include <iostream>

void ms_blind_deconv(cv::Mat& blurredImage, const BlindDeblurOptions& opts,
		cv::Mat& kernelImage, cv::Mat& deblurredImage) {

	//
	// Do multi-scale blind deconvolution given input file name and options
	// structure opts. Returns a double deblurred image along with estimated
	// kernel. Following the kernel estimation, a non-blind deconvolution is run.
	//
	// Copyright (2011): Dilip Krishnan, Rob Fergus, New York University.
	//

	cv::Mat y;

	if (blurredImage.empty()) {
		std::cout << "No image provided in blurredImage!!!" << std::endl;
		return;
	} else {
		// TODO: Check the number of the channels!
		switch (blurredImage.channels()) {
		case 1:
			y.create(blurredImage.rows, blurredImage.cols, CV_32FC1);
			blurredImage.convertTo(y, CV_32FC1, 1.0 / 255.0, 0.0);
			break;
		case 3:
			y.create(blurredImage.rows, blurredImage.cols, CV_32FC3);
			blurredImage.convertTo(y, CV_32FC3, 1.0 / 255.0, 0.0);
			break;
		default:
			std::cerr << "Algorithm not implemented for "
					<< blurredImage.channels() << " !" << std::endl;
			return;
		}
	}

	// prescale the image if it's too big; kernel size is defined for the SCALED image
	cv::Mat y1;
	cv::resize(y, y1, cv::Size(), opts.prescale, opts.prescale,
			cv::INTER_LINEAR);
	y = y1;

	// save off for non-blind deconvolution
	cv::Mat yorig = y.clone();

	// gamma correct
	cv::pow(y, opts.gamma_correct, y);

	// use a window to estimate kernel
	if (opts.kernel_est_win.area() == 0) {
		if (y.channels() == 3) {
			cv::Mat tmpImage;
			cv::cvtColor(y, tmpImage, cv::COLOR_RGB2GRAY);
			cv::cvtColor(tmpImage, y, cv::COLOR_GRAY2RGB);
		}
	} else {
		if (y.channels() == 3) {
			cv::Mat tmpImage = y(opts.kernel_est_win).clone();
			cv::cvtColor(tmpImage, tmpImage, cv::COLOR_RGB2GRAY);
			cv::cvtColor(tmpImage, y, cv::COLOR_GRAY2RGB);
		}
	}

	cv::Mat b(opts.kernel_size, opts.kernel_size, CV_32FC1);

	int bhs = b.rows / 2;

	// set kernel size for coarsest level - must be odd
	int minsize = std::max(3.0,
			2.0 * floor(((opts.kernel_size - 1.0) / 16.0)) + 1.0);
	std::cout << "Kernel size at coarsest level is " << minsize << std::endl;

	// l2 norm of gradient images
	float l2norm = 6;
	float resize_step = sqrt(2.0);

	// determine number of scales
	int num_scales = 1;
	int tmp0 = minsize;
	std::vector<int> ksize;
	while (tmp0 < opts.kernel_size) {
		ksize.push_back(tmp0);
		num_scales = num_scales + 1;
		tmp0 = ceil(tmp0 * resize_step);
		if ((tmp0 % 2) == 0) {
			tmp0 = tmp0 + 1;
		};
	}
	ksize.push_back(opts.kernel_size);

	std::vector<cv::Mat> ks(num_scales);
	std::vector<cv::Mat> ls(num_scales);

	int error_flag = 0;

	// blind deconvolution - multiscale processing
	for (int s = 0; s < num_scales; s++) {

		int k1, k2;

		if (s == 0) {
			// at coarsest level, initialize kernel
			init_kernel(ksize[0], ks[s]);
			k1 = ksize[0];
			k2 = k1; // always square kernel assumed
		} else {
			// upsample kernel from previous level to next finer level
			k1 = ksize[s];
			k2 = k1; // always square kernel assumed

			// resize kernel from previous level
			cv::Mat tmp = ks[s - 1];
			cv::threshold(tmp, tmp, 0.0, 1.0, cv::THRESH_TOZERO);
			cv::Scalar sumTmp = cv::sum(tmp);
			tmp = tmp / sumTmp.val[0];
			cv::resize(tmp, ks[s], cv::Size(k1, k2), 0.0, 0.0,
					cv::INTER_LINEAR);
			// bilinear interpolantion not guaranteed to sum to 1 - so renormalize
			cv::threshold(ks[s], ks[s], 0.0, 1.0, cv::THRESH_TOZERO);
			cv::Scalar sumk = cv::sum(ks[s]);
			ks[s] = ks[s] / sumk.val[0];
		}

		// image size at this level
		int r = floor(y.rows * k1 / b.rows);
		int c = floor(y.cols * k2 / b.rows);

		if (s == num_scales - 1) {
			r = y.rows;
			c = y.cols;
		}

		std::cout << "Processing scale " << s << "/" << num_scales
				<< "; kernel size " << k1 << "x" << k2 << "; image size " << c
				<< "x" << r << std::endl;

		// resize y according to the ratio of filter sizes
		cv::Mat ys;
		cv::resize(y, ys, cv::Size(c, r), 0.0, 0.0, cv::INTER_LINEAR);
		cv::cvtColor(ys, ys, cv::COLOR_RGB2GRAY);
		cv::Mat yx, yy;

//		cv::Mat dx(1, 3, CV_32FC1);
//		dx.at<float>(cv::Point(0, 0)) = 1.0;
//		dx.at<float>(cv::Point(1, 0)) = 0.0;
//		dx.at<float>(cv::Point(2, 0)) = -1.0;
//		cv::filter2D(ys, yx, -1, dx);

//		cv::Mat dy(3, 1, CV_32FC1);
//		dy.at<float>(cv::Point(0, 0)) = 1.0;
//		dy.at<float>(cv::Point(0, 1)) = 0.0;
//		dy.at<float>(cv::Point(0, 2)) = -1.0;
//		cv::filter2D(ys, yy, -1, dy);

		cv::Sobel(ys, yx, CV_32F, 1, 0);
		cv::Sobel(ys, yy, CV_32F, 0, 1);

//		cv::Laplacian(ys, yx, CV_32F, 5);
//		cv::Laplacian(ys, yy, CV_32F, 5);

		c = std::min(yx.cols, yy.cols);
		r = std::min(yx.rows, yy.rows);

		cv::Mat gArray[2];
		gArray[0] = yx;
		gArray[1] = yy;
		cv::Mat g;
		cv::merge(gArray, 2, g);

		// normalize to have l2 norm of a certain size
		yx = yx * l2norm / cv::norm(yx);
		yy = yy * l2norm / cv::norm(yy);

		cv::merge(gArray, 2, g);

		if (s == 0) {
			ls[0] = g;
		} else {
			if (error_flag != 0) {
				ls[s] = g;
			} else {
				// upscale the estimated derivative image from previous level
				cv::Mat tmp1(ls[s - 1].rows, ls[s - 1].cols, CV_32FC1);
				cv::Mat tmp2(ls[s - 1].rows, ls[s - 1].cols, CV_32FC1);
				std::vector<cv::Mat> tmps(2);
				tmps[0] = tmp1;
				tmps[1] = tmp2;
				cv::split(ls[s - 1], tmps);

				cv::Mat tmp1_up;
				cv::Mat tmp2_up;
				cv::resize(tmp1, tmp1_up, cv::Size(c, r), 0.0, 0.0,
						cv::INTER_LINEAR);
				cv::resize(tmp2, tmp2_up, cv::Size(c, r), 0.0, 0.0,
						cv::INTER_LINEAR);

				tmps[0] = tmp1_up;
				tmps[1] = tmp2_up;
				cv::merge(tmps, ls[s]);
			}
		}

		cv::Mat tmp1(ls[s].rows, ls[s].cols, CV_32FC1);
		cv::Mat tmp2(ls[s].rows, ls[s].cols, CV_32FC1);
		std::vector<cv::Mat> tmps(2);
		tmps[0] = tmp1;
		tmps[1] = tmp2;
		cv::split(ls[s], tmps);

		tmps[0] = tmps[0] * l2norm / cv::norm(tmps[0]);
		tmps[1] = tmps[1] * l2norm / cv::norm(tmps[1]);

		cv::merge(tmps, ls[s]);

		// call kernel estimation for this scale
		opts.lambda = opts.min_lambda;

		error_flag = ss_blind_deconv(g, ls[s], ks[s], opts.lambda, opts);

		// TODO: Kernel test
		cv::Mat testDeblur = ys.clone();

		cgDeblur(ys, ks[s], testDeblur);
		rlDeblur(ys, ks[s], testDeblur);
		// cv::namedWindow("test deblur", cv::WINDOW_NORMAL);
		// cv::imshow("test deblur", testDeblur);
		// cv::waitKey(10);

		if (error_flag < 0) {
			ks[s] = cv::Scalar(0.0);
			ks[s].at<float>((ks[s].rows + 0) / 2, (ks[s].cols + 0) / 2) = 1.0;

			std::cout
					<< "Bad error - just set output to delta kernel and return"
					<< std::endl;
		}

		// center the kernel
		int c1 = ls[s].rows / 2;

		tmp1.copyTo(tmps[0]);
		tmp2.copyTo(tmps[1]);
//		cv::split(ls[s], tmps);
//
//		cv::Mat tmp1_shifted, tmp2_shifted;
//
//		center_kernel_separate(tmp1, tmp2, ks[s], tmp1_shifted, tmp2_shifted,
//				ks[s]);
//
//		tmp1_shifted.copyTo(tmps[0]);
//		tmp2_shifted.copyTo(tmps[1]);
//		cv::merge(tmps, ls[s]);

		// set elements below threshold to 0
		if (s == num_scales - 1) {
			kernelImage = ks[s];
			double minVal;
			double maxVal;
			cv::minMaxLoc(kernelImage, &minVal, &maxVal);
			cv::threshold(kernelImage, kernelImage, opts.k_thresh * maxVal, 1.0,
					cv::THRESH_TOZERO);
			cv::Scalar sumTmp = cv::sum(kernelImage);
			if (sumTmp.val[0] != 0) {
				kernelImage = kernelImage / sumTmp.val[0];
			} else {
				//TODO: debug print
				std::cout << "PROBLEM!" << std::endl;
			}

		}
	}

	int padsize = bhs;

	cv::Mat ycbcr;
	if (opts.use_ycbcr) {
		cv::cvtColor(yorig, ycbcr, cv::COLOR_RGB2YCrCb);
//		opts.nb_alpha = 1;
	}

	cv::Mat ypad;
	int borderType = cv::BORDER_REPLICATE;
	int top, bottom, left, right;
	top = (int) (kernelImage.rows);
	bottom = (int) (kernelImage.rows);
	left = (int) (kernelImage.cols);
	right = (int) (kernelImage.cols);

	if (opts.use_ycbcr) {
		copyMakeBorder(ycbcr, ypad, top, bottom, left, right, borderType,
				cv::Scalar(0));
	} else {
		copyMakeBorder(yorig, ypad, top, bottom, left, right, borderType,
				cv::Scalar(0));
	}

	cv::Mat tmp;
	fast_deconv_bregman(ypad, kernelImage, opts.nb_lambda, opts.nb_alpha, tmp);

	// TODO: Debug print
//	cv::imshow("ycbcr", ycbcr);
//	cv::imshow("tmp", tmp);

	tmp(cv::Rect(kernelImage.cols, kernelImage.rows, ycbcr.cols, ycbcr.rows)).copyTo(
			deblurredImage);

	if (opts.use_ycbcr) {
		cv::cvtColor(deblurredImage, deblurredImage, cv::COLOR_YCrCb2RGB);
	}


	// TODO: Kernel test
//	cgDeblur(yorig, kernelImage, deblurredImage);
//	rlDeblur(yorig, kernelImage, deblurredImage);

	//TODO: Debug print
	cv::imshow("Blurred", y);
	cv::imshow("Deblurred", deblurredImage);
	cv::imshow("Kernel", kernelImage);
}

void init_kernel(int minsize, cv::Mat& k) {
	k.create(minsize, minsize, CV_32FC1);
	k = cv::Scalar(0);
	k(cv::Rect((minsize - 1) / 2, (minsize - 1) / 2, 1, 1)) = cv::Scalar(
			1.0 / 2.0);
}

// Center the kernel by translation so that boundary issues are mitigated. Additionally,
// if one shifts the kernel the the image must also be shifted in the
// opposite direction.
void center_kernel_separate(const cv::Mat& x, const cv::Mat& y,
		const cv::Mat& k, cv::Mat& x_shifted, cv::Mat& y_shifted,
		cv::Mat& k_shifted) {

	// get centre of mass
	double sumX = 0;
	double sumY = 0;
	double sum = 0;
	for (int i = 0; i < k.rows; i++) {
		for (int j = 0; j < k.cols; j++) {
			float pixel = k.at<float>(i, j);
			sumX += i * pixel;
			sumY += j * pixel;
			sum += pixel;
		}
	}

	//TODO: Debug print!
	std::cout << "sum: " << sum << "  sumX: " << sumX << "  sumY: " << sumY
			<< std::endl;

	double mu_y;
	double mu_x;

	if (sum != 0) {
		mu_y = sumY / sum;
		mu_x = sumX / sum;
	} else {
		mu_y = (k.rows + 1) / 2;
		mu_x = (k.cols + 1) / 2;
	}

	// get mean offset
	double offset_x = -round(floor(k.cols / 2.0) - mu_x);
	double offset_y = -round(floor(k.rows / 2.0) - mu_y);

	std::cout << "CenterKernel: weightedMean[" << mu_x - 1 << " " << mu_y - 1
			<< "] offset[" << offset_x << " " << offset_y << "]" << std::endl;

	// make kernel to do translation
	cv::Mat shift_kernel(abs(offset_y * 2) + 1, abs(offset_x * 2) + 1,
	CV_32FC1);
	shift_kernel = cv::Scalar(0.0);
	shift_kernel.at<float>(
			cv::Point(abs(offset_x) + offset_x, abs(offset_y) + offset_y)) =
			1.0;

	// shift both image and blur kernel
	cv::Mat kshift = k.clone();
	kshift = cv::Scalar(0);

	for (int row = 0; row < k.rows; row++) {
		int shifterRow = row - offset_y;
		if (shifterRow < 0 || shifterRow > k.rows - 1) {
			continue;
		}
		for (int col = 0; col < k.cols; col++) {
			int shiftedCol = col - offset_x;
			if (shiftedCol < 0 || shiftedCol > k.cols - 1) {
				continue;
			}
			kshift.at<float>(cv::Point(shiftedCol, shifterRow)) = k.at<float>(
					cv::Point(col, row));
		}
	}

	kshift.copyTo(k_shifted);

	// TODO: Debug print
//	cv::namedWindow("kshift", cv::WINDOW_NORMAL);
//	cv::namedWindow("k1", cv::WINDOW_NORMAL);
//	cv::imshow("kshift", kshift);
//	cv::imshow("k1", k);
//	cv::waitKey(10);

//	cv::Mat flippedKs;
//	cv::flip(shift_kernel, flippedKs, -1);

	cv::Mat xshift = x.clone();
	xshift = cv::Scalar(0);
//	cv::filter2D(x, xshift, -1, flippedKs);
//	tmp1_shifted = xshift;

	cv::Mat yshift = y.clone();
	yshift = cv::Scalar(0);
//	cv::filter2D(y, yshift, -1, flippedKs);
//	tmp2_shifted = yshift;

	for (int row = 0; row < x.rows; row++) {
		int shifterRow = row + offset_y;
		if (shifterRow < 0 || shifterRow > x.rows - 1) {
			continue;
		}
		for (int col = 0; col < x.cols; col++) {
			int shiftedCol = col + offset_x;
			if (shiftedCol < 0 || shiftedCol > x.cols - 1) {
				continue;
			}
			xshift.at<float>(cv::Point(shiftedCol, shifterRow)) = x.at<float>(
					cv::Point(col, row));
			yshift.at<float>(cv::Point(shiftedCol, shifterRow)) = y.at<float>(
					cv::Point(col, row));
		}
	}

	xshift.copyTo(x_shifted);
	yshift.copyTo(y_shifted);

}

void cgDeblur(const cv::Mat& blurredImage, const cv::Mat& kernel,
		cv::Mat& deblurredImage) {

	cv::Mat flippedBlurKernel;
	cv::flip(kernel, flippedBlurKernel, -1);

	cv::Mat residualImage; // residual image
	cv::Mat preconditionedImage; // preconditioned image
	cv::Mat blurredPreconditionedImage; // blurred preconditioned image
	cv::Mat differenceResidualImage; // temp vector
	cv::Mat regularizationImage = blurredImage.clone();

	double preconditionWeight, updateWeight, residualNorm, initialNorm;
	double regularizationWeight = 0.01;
	int it; // Iteration counter

	// initial approximation of the restored image
	deblurredImage = blurredImage.clone();

	// initial approximation of the residual
	residualImage = blurredImage.clone();
	cv::filter2D(deblurredImage, residualImage, CV_32FC1, kernel);

	cv::subtract(blurredImage, residualImage, differenceResidualImage);
	cv::filter2D(differenceResidualImage, residualImage, CV_32FC1,
			flippedBlurKernel);
	// Add regularization
//	cv::Laplacian(deblurredImage, differenceResidualImage, CV_32FC1);
//	cv::Laplacian(differenceResidualImage, regularizationImage, CV_32FC1);
	deblurredImage.copyTo(regularizationImage);
	cv::addWeighted(regularizationImage, regularizationWeight, residualImage,
			1.0, 0.0, residualImage);

	initialNorm = sqrt(cv::norm(residualImage, cv::NORM_L2))
			/ (residualImage.cols * residualImage.rows);

	//initial approximation of preconditioner
	preconditionedImage = residualImage.clone();

	//initial approximation of preconditioned blurred image
	blurredPreconditionedImage = preconditionedImage.clone();
	cv::filter2D(preconditionedImage, differenceResidualImage, CV_32FC1,
			kernel);
	cv::filter2D(differenceResidualImage, blurredPreconditionedImage,
	CV_32FC1, flippedBlurKernel);
	// Add regularization
	//	cv::Laplacian(preconditionedImage, differenceResidualImage, CV_32FC1);
	//	cv::Laplacian(differenceResidualImage, regularizationImage, CV_32FC1);
	preconditionedImage.copyTo(regularizationImage);
	cv::addWeighted(regularizationImage, regularizationWeight,
			blurredPreconditionedImage, 1.0, 0.0, blurredPreconditionedImage);

	differenceResidualImage = residualImage.clone();
	differenceResidualImage = cv::Scalar(0);

	double bestNorm = initialNorm;
	cv::Mat bestRestoredImage = deblurredImage.clone();

	//reset iteration counter
	it = 0;

	do {
		// beta_k first part
		preconditionWeight = residualImage.dot(residualImage);

		//alpha_k
		updateWeight = preconditionWeight
				/ preconditionedImage.dot(blurredPreconditionedImage);

		//x_k
		cv::addWeighted(deblurredImage, 1.0, preconditionedImage, updateWeight,
				0.0, deblurredImage);

		//r_k
		residualImage.copyTo(differenceResidualImage);
		cv::addWeighted(residualImage, 1.0, blurredPreconditionedImage,
				-updateWeight, 0.0, residualImage);
		cv::subtract(residualImage, differenceResidualImage,
				differenceResidualImage);

		//norm calculation
		residualNorm = sqrt(cv::norm(residualImage, cv::NORM_L2))
				/ (residualImage.cols * residualImage.rows);

		//beta_k second part
		preconditionWeight = residualImage.dot(differenceResidualImage)
				/ preconditionWeight;

		//p_k
		cv::addWeighted(residualImage, 1.0, preconditionedImage,
				1.0 * preconditionWeight, 0.0, preconditionedImage);

		//Ap_k
		cv::filter2D(preconditionedImage, differenceResidualImage, CV_32FC1,
				kernel);
		cv::filter2D(differenceResidualImage, blurredPreconditionedImage,
		CV_32FC1, flippedBlurKernel);
		// Add regularization
		//		cv::Laplacian(preconditionedImage, differenceResidualImage, CV_32FC1);
		//		cv::Laplacian(differenceResidualImage, regularizationImage, CV_32FC1);
		preconditionedImage.copyTo(regularizationImage);
		cv::addWeighted(regularizationImage, regularizationWeight,
				blurredPreconditionedImage, 1.0, 0.0,
				blurredPreconditionedImage);

		if (residualNorm < bestNorm) {
			bestNorm = residualNorm;
			deblurredImage.copyTo(bestRestoredImage);
		}

		cv::imshow("CG Deblurred", deblurredImage);

		char c = cv::waitKey(10);
		if (c == 27)
			break;

		//		std::cout << " Iteration: " << it << " Norm: " << residualNorm
		//				<< std::endl;

		// std::cout << " Iteration: " << it << " Norm: " << residualNorm
		// 		<< " preconditionWeight: " << preconditionWeight
		// 		<< " updateWeight: " << updateWeight << std::endl;

		it++;
	} while ((residualNorm > 5e-7) && (it < 201));

	//	imgx.convertTo(deblurredImage, CV_8UC1, 255.0, 0.0);
	bestRestoredImage.copyTo(deblurredImage);
	std::cout << "Best norm: " << bestNorm << std::endl;
}

void rlDeblur(const cv::Mat& blurredImage, const cv::Mat& kernel,
		cv::Mat& deblurredImage) {

	cv::Mat flippedBlurKernel;
	cv::flip(kernel, flippedBlurKernel, -1);

	float residualNorm;  // norm of the images

	// Initial approximation of blurred image.
	cv::filter2D(deblurredImage, blurredImage, CV_32FC1, flippedBlurKernel);

	cv::Mat reblurredImage = blurredImage.clone();
	cv::Mat reblurredTransposedImage = blurredImage.clone();
	cv::Mat weightImage = blurredImage.clone();

	double regularizationWeight = 0.001;
	cv::Mat regularizationImage = blurredImage.clone();
	cv::Mat regularizationImageTransposed = blurredImage.clone();

	//Richardson-Lucy starts here
	int iteration = 0;
	do {

		cv::filter2D(deblurredImage, reblurredImage, CV_32FC1, kernel);
		cv::filter2D(reblurredImage, reblurredTransposedImage, CV_32FC1,
				flippedBlurKernel);

		// Add regularization
		cv::Laplacian(deblurredImage, regularizationImageTransposed,
		CV_32FC1);
		cv::Laplacian(regularizationImageTransposed, regularizationImage,
		CV_32FC1);
//		deblurredImage.copyTo(regularizationImage);
		cv::addWeighted(regularizationImage, regularizationWeight,
				reblurredTransposedImage, 1.0, 0.0, reblurredTransposedImage);

		cv::divide(blurredImage, reblurredTransposedImage, weightImage);

		cv::Mat oldDeblurredImage = deblurredImage.clone();

		// pixel by pixel multiply
		cv::multiply(weightImage, deblurredImage, deblurredImage);

		residualNorm = sqrt(
				cv::norm(deblurredImage - oldDeblurredImage, cv::NORM_L2))
				/ (blurredImage.cols * blurredImage.rows);

		// std::cout << iteration << "  " << residualNorm << std::endl;

		cv::imshow("Deblurred", deblurredImage);
		char c = cv::waitKey(10);

		if (c == 27)
			break;

		iteration++;
	} while ((residualNorm > 5e-7) && (iteration < 1301));
}
