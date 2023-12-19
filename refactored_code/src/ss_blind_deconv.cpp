/*
 * ss_blind_deconv.cpp
 *
 *  Created on: Jan 12, 2014
 *      Author: vladiant
 */

#include "ss_blind_deconv.h"
#include "pcg_kernel_irls_conv.h"

//TODO: debug print
#include <iostream>

int ss_blind_deconv(cv::Mat& y, cv::Mat& x, cv::Mat& k, float lambda,
		const BlindDeblurOptions& opts) {

	//
	// Do single-scale blind deconvolution using the input initializations
	// x and k. The cost function being minimized is: min_{x,k}
	// \lambda/2 |y - x \oplus k|^2 + |x|_1/|x|_2 + k_reg_wt*|k|_1
	//

	int error_flag = 0;

	cv::Mat k_init = k.clone();
	int khs = floor(k.cols / 2.0);

	int m = y.rows;
	int n = y.cols;
	int k1 = k.rows;
	int k2 = k.cols;

	int m2 = n / 2;

	// arrays to hold costs
	std::vector<float> lcost;
	std::vector<float> pcost;
	unsigned int totiter = 0;

	// Split y into 2 parts: x and y gradients; handle independently throughout
	cv::Mat y11(y.rows, y.cols, CV_32FC1);
	cv::Mat y12(y.rows, y.cols, CV_32FC1);
	std::vector<cv::Mat> y1(2);
	y1[0] = y11;
	y1[1] = y12;
	cv::split(y, y1);

	cv::Mat y21 =
			y11(
					cv::Rect(khs, khs, y11.cols - 2 * khs - 1,
							y11.rows - 2 * khs - 1)).clone();
	cv::Mat y22 =
			y12(
					cv::Rect(khs, khs, y12.cols - 2 * khs - 1,
							y12.rows - 2 * khs - 1)).clone();
	std::vector<cv::Mat> y2(2);
	y2[0] = y21;
	y2[1] = y22;

	cv::Mat x11(x.rows, x.cols, CV_32FC1);
	cv::Mat x12(x.rows, x.cols, CV_32FC1);
	std::vector<cv::Mat> x1(2);
	x1[0] = x11;
	x1[1] = x12;
	cv::split(x, x1);

	cv::Mat tmp11(x.rows, x.cols, CV_32FC1);
	cv::Mat tmp12(x.rows, x.cols, CV_32FC1);

	cv::filter2D(x11, tmp11, -1, k);
	cv::filter2D(x12, tmp12, -1, k);

	tmp11 = tmp11 - y11;
	tmp12 = tmp12 - y12;

	double normTmp11 = cv::norm(tmp11, cv::NORM_L2);
	double normTmp12 = cv::norm(tmp12, cv::NORM_L2);

	lcost.push_back(
			(lambda / 2.0) * (normTmp11 * normTmp11 + normTmp12 * normTmp12));

	pcost.push_back(
			cv::norm(x11, cv::NORM_L1) / cv::norm(x11, cv::NORM_L2)
					+ cv::norm(x12, cv::NORM_L1) / cv::norm(x12, cv::NORM_L2));

	double normy[2];
	normy[0] = cv::norm(y1[0], cv::NORM_L2);
	normy[1] = cv::norm(y1[1], cv::NORM_L2);

	// mask
	cv::Mat mask1(x11.rows, x11.cols, CV_32FC1);
	cv::Mat mask2(x12.rows, x12.cols, CV_32FC1);
	mask1 = cv::Scalar(1.0);
	mask2 = cv::Scalar(1.0);

	float lambda_orig = lambda;
	float delta_orig = opts.delta;

	// x update step
	for (int iter = 0; iter < opts.xk_iter; iter++) {
		lambda = lambda_orig; // /(1.15^(xk_iter-iter)); // seems to work better
							  // without this

		bool skip_rest = false;

		int totiter_before_x = totiter;

		float cost_before_x = lcost[totiter] + pcost[totiter];
		std::vector<cv::Mat> x2(2);
		x1[0].copyTo(x2[0]);
		x1[1].copyTo(x2[1]);

		float delta = opts.delta;
		while (delta > 1e-4) {

			// TODO: Debug print
//			std::cout << "delta: " << delta << std::endl;

			for (int out_iter = 0; out_iter < opts.x_out_iter; out_iter++) {

				if (skip_rest) {
					break;
				}

				double normx1 = cv::norm(x11, cv::NORM_L2);
				double normx2 = cv::norm(x12, cv::NORM_L2);

				double beta1 = lambda * normx1;
				double beta2 = lambda * normx2;

				// TODO: Debug print
//				std::cout << "out_iter: " << out_iter << std::endl;
//				std::cout << "normx1: " << normx1 << std::endl;
//				std::cout << "normx2: " << normx2 << std::endl;
//				std::cout << "beta1: " << beta1 << std::endl;
//				std::cout << "beta2: " << beta2 << std::endl;

				for (int inn_iter = 1; inn_iter < opts.x_in_iter; inn_iter++) {

					if (skip_rest) {
						break;
					}

					totiter++;
					opts.lambda = lambda;
					std::vector<cv::Mat> x1prev(2);
					x1[0].copyTo(x1prev[0]);
					x1[1].copyTo(x1prev[1]);

					cv::Mat flippedK;
					cv::flip(k, flippedK, -1);

					cv::Mat tempV;

					cv::Mat tempV1;
					cv::filter2D(x1prev[0], tempV1, -1, k);
					cv::subtract(y11, tempV1, tempV);
					cv::multiply(tempV, mask1, tempV1);
					cv::filter2D(tempV1, tempV, -1, flippedK);
					cv::Mat v1 = x1prev[0] + beta1 * delta * tempV;


					cv::Mat tempV2;
					cv::filter2D(x1prev[1], tempV2, -1, k);
					cv::subtract(y12, tempV2, tempV);
					cv::multiply(tempV, mask2, tempV2);
					cv::filter2D(tempV2, tempV, -1, flippedK);
					cv::Mat v2 = x1prev[1] + beta2 * delta * tempV;

					cv::max(cv::Mat(cv::abs(v1) - cv::Scalar(delta)), 0, x11);

					// sign
					cv::multiply(x11, cv::abs(v1), x11);
					cv::divide(x11, v1, x11);
					cv::patchNaNs(x11, 0);

					cv::max(cv::Mat(cv::abs(v2) - cv::Scalar(delta)), 0, x12);
					// sign
					cv::multiply(x12, cv::abs(v2), x12);
					cv::divide(x12, v2, x12);
					cv::patchNaNs(x12, 0); 

					cv::filter2D(x11, tmp11, -1, k);
					cv::subtract(tmp11, y11, tmp11);

					cv::filter2D(x12, tmp12, -1, k);
					cv::subtract(tmp12, y12, tmp12);

					double normTmp11 = cv::norm(tmp11, cv::NORM_L2);
					double normTmp12 = cv::norm(tmp12, cv::NORM_L2);
					if (totiter < lcost.size() - 1) {
						lcost[totiter] =
								(lambda / 2.0)
										* (normTmp11 * normTmp11
												+ normTmp12 * normTmp12);
					} else {
						lcost.push_back(
								(lambda / 2.0)
										* (normTmp11 * normTmp11
												+ normTmp12 * normTmp12));
					} // if (totiter < lcost.size() - 1)

					if (totiter < pcost.size() - 1) {
						pcost[totiter] = cv::norm(x11, cv::NORM_L1)
								/ cv::norm(x11)
								+ cv::norm(x12, cv::NORM_L1) / cv::norm(x12);
					} else {
						pcost.push_back(
								cv::norm(x11, cv::NORM_L1) / cv::norm(x11)
										+ cv::norm(x12, cv::NORM_L1)
												/ cv::norm(x12));
					} //if (totiter < pcost.size() - 1) ..

					// TODO: Debug print
//					std::cout << "lcost[" << totiter << "]: " << lcost[totiter]
//							<< std::endl;
//					std::cout << "pcost[" << totiter << "]: " << pcost[totiter]
//							<< std::endl;

					//TODO: Debug print
//					cv::namedWindow("test x11", cv::WINDOW_NORMAL);
//					cv::namedWindow("test x12", cv::WINDOW_NORMAL);
//					cv::imshow("test x11", x11);
//					cv::imshow("test x12", x12);
//					cv::waitKey(10);

				} // for (int inn_iter = ..
			} // for (int out_iter ..

			// prevent blow up in costs due to going uphill; since l1/l2 is so
			// nonconvex it is sometimes difficult to prevent going uphill crazily.
			float cost_after_x = lcost[totiter] + pcost[totiter];

			// TODO: Debug print
//			std::cout << "cost_after_x: " << cost_after_x << std::endl;

			if (cost_after_x > 3 * cost_before_x) {
				totiter = totiter_before_x;
				x2[0].copyTo(x1[0]);
				x2[1].copyTo(x1[1]);
				delta = delta / 2;
			} else {
				break;
			} // if (cost_after_x ...

		} // while (delta > 1e-4) ...

		opts.delta = delta;

		// set up options for the kernel estimation
		opts.lambda = opts.k_reg_wt;
		opts.pcg_tol = 1e-4;
		opts.pcg_its = 1;

		cv::Mat k_prev = k.clone();
		pcg_kernel_irls_conv(k_prev, x1, y2, opts, k); // using conv2's
		cv::threshold(k, k, 0.0, 1.0, cv::THRESH_TOZERO);
		cv::Scalar sumk = cv::sum(k);
		k = k / sumk.val[0];
	} //for (int iter = 0 ...

	// combine back into output
	cv::merge(x1, x);

	//TODO: Debug print
	cv::namedWindow("test kernel", cv::WINDOW_NORMAL);
	double kernelMin, kernelMax;
	cv::minMaxLoc(k, &kernelMin, &kernelMax);
	cv::imshow("test kernel", k * (1.0 / kernelMax));
//	cv::imshow("test x1", x11);
//	cv::imshow("test x2", x12);
	cv::waitKey(10);

	return error_flag;
}
