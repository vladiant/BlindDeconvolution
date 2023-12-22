/*
 * pcg_kernel_irls_conv.cpp
 *
 *  Created on: Jan 19, 2014
 *      Author: vladiant
 */

#include "pcg_kernel_irls_conv.h"

#include <iostream>

#include "conv2.h"

//
// Use Iterative Re-weighted Least Squares to solve l_1 regularized kernel
// update with sum to 1 and nonnegativity constraints. The problem that is
// being minimized is:
//
// min 1/2\|Xk - Y\|^2 + \lambda \|k\|_1
//
// Inputs:
// k_init = initial kernel, or scalar specifying size
// X = sharp image
// Y = blurry image
// opts = options (see below)
//
// Outputs:
// k_out = output kernel
//
// This version of the function uses spatial convolutions. Everything is maintained as 2D arrays
//

void pcg_kernel_irls_conv(const cv::Mat& k_init, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>& Y, const BlindDeblurOptions& opts, const BlindDeblurContext& aContext,
		cv::Mat& k_out) {

	// Defaults
	if (false) { //nargin == 3
//		opts.lambda = 0;
//		// PCG parameters
//		opts.pcg_tol = 1e-8;
//		opts.pcg_its = 100;
//		std::cout
//				<< "Input options not defined - really no reg/constraints on the kernel?"
//				<< std::endl;
	}

	float lambda = aContext.lambda;
	float pcg_tol = aContext.pcg_tol;
	int pcg_its = aContext.pcg_its;

	// assume square kernel
	int ks = k_init.cols;
	int ks2 = ks / 2;

	// precompute RHS
	std::vector<cv::Mat> flipX(X.size());
	std::vector<cv::Mat> rhs(X.size());

    Convolver conv;
	for (unsigned int i = 0; i < X.size(); i++) {

		rhs[i].create(ks, ks, CV_32FC1);

		cv::flip(X[i], flipX[i], -1);

		// precompute X^T Y term on RHS (e = ks^2 length vector of all 1's)

		/*
		 cv::Mat tempX = X[i].clone();
		 cv::dft(tempX, tempX);

		 cv::Mat tempY = X[i].clone();
		 tempY = cv::Scalar(0);

		 for (int row = 0; row < Y[i].rows; row++) {
		 for (int col = 0; col < Y[i].cols; col++) {
		 tempY.at<float>(cv::Point(col, row)) = Y[i].at<float>(
		 cv::Point(col, row));
		 }
		 }

		 cv::dft(tempY, tempY);
		 cv::Mat tempRhs = X[i].clone();

		 cv::mulSpectrums(tempY, tempX, tempRhs, cv::DFT_COMPLEX_OUTPUT, true);
		 cv::dft(tempRhs, tempRhs,
		 cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

		 copyImageToKernel((float*) tempRhs.data, tempRhs.cols, tempRhs.rows,
		 tempRhs.cols, (float*) rhs[i].data, rhs[i].cols, rhs[i].rows,
		 rhs[i].cols);

		 */

		conv((float*) flipX[i].data, flipX[i].cols, flipX[i].rows,
				flipX[i].cols, (float*) Y[i].data, Y[i].cols, Y[i].rows,
				Y[i].cols, (float*) rhs[i].data, rhs[i].cols, rhs[i].rows,
				rhs[i].cols);

		// TODO: Debug print
//		cv::namedWindow("Y[i]", cv::WINDOW_NORMAL);
//		cv::imshow("Y[i]", Y[i]);
//		cv::namedWindow("X[i]", cv::WINDOW_NORMAL);
//		cv::imshow("X[i]", X[i]);
//		cv::namedWindow("tempRhs", cv::WINDOW_NORMAL);
//		cv::imshow("tempRhs", 0.01 * tempRhs);
//		cv::namedWindow("rhs[i]", cv::WINDOW_NORMAL);
//		cv::imshow("rhs[i]", 0.01 * rhs[i]);
//		cv::waitKey(10);
	}

	cv::Mat tmp(rhs[0].rows, rhs[0].cols, CV_32FC1);
	tmp = cv::Scalar(0.0);
	for (unsigned int i = 0; i < X.size(); i++) {
		tmp = tmp + rhs[i];
	}
	rhs[0] = tmp;

	//TODO: Debug print
//	cv::namedWindow("rhs[0]", cv::WINDOW_NORMAL);
//	cv::imshow("rhs[0]", 0.01*rhs[0]);
//	cv::waitKey(10);

	k_init.copyTo(k_out);

	// Set exponent for regularization
	float exp_a = 1.0;

	// outer loop
	for (int iter = 0; iter < 1; iter++) {

		cv::Mat k_prev = k_out;
		// compute diagonal weights for IRLS
		cv::Mat weights_l1 = cv::abs(k_prev);
		weights_l1 = cv::max(weights_l1, 0.0001);
		cv::pow(weights_l1, exp_a - 2, weights_l1);
		weights_l1 = lambda * weights_l1;

		//TODO: Debug print
//		cv::namedWindow("weights_l1", cv::WINDOW_NORMAL);
//		cv::imshow("weights_l1", 1 * weights_l1);
//		cv::waitKey(10);

		local_cg(k_prev, X, flipX, ks, weights_l1, rhs[0], pcg_tol, pcg_its,
				k_out);

		//TODO: Debug print
//		cv::namedWindow("k_out", cv::WINDOW_NORMAL);
//		cv::imshow("k_out", k_out.rows * k_out.cols * k_out / 10);
//		cv::waitKey(10);
	}
}

// local implementation of CG to solve the reweighted least squares problem
void local_cg(const cv::Mat& k, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>& flipX, int ks, const cv::Mat& weights_l1,
		const cv::Mat& rhs, float tol, int max_its, cv::Mat& k_out) {

	k_out = k.clone();
	cv::Mat Ak;
	pcg_kernel_core_irls_conv(k, X, flipX, ks, weights_l1, Ak);

	cv::Mat r = rhs - Ak;

	//TODO: Debug print
//	cv::namedWindow("Ak", cv::WINDOW_NORMAL);
//	cv::imshow("Ak", 0.1 * Ak);
//	cv::namedWindow("r0", cv::WINDOW_NORMAL);
//	cv::imshow("r0", 0.1 * r);
//	cv::waitKey(10);

	for (int iter = 0; iter < max_its; iter++) {
		float rho = r.dot(r);

		float rho_1 = 0.0;
		cv::Mat p(r.rows, r.cols, CV_32FC1);

		if (iter > 0) {
			float beta = rho / rho_1;
			p = r + beta * p;
		} else {
			r.copyTo(p);
		}

		cv::Mat Ap = p.clone();
		pcg_kernel_core_irls_conv(p, X, flipX, ks, weights_l1, Ap);

		//TODO: Debug print
//		cv::namedWindow("Ap", cv::WINDOW_NORMAL);
//		cv::imshow("Ap", 0.1 * Ap);

		cv::Mat q = Ap.clone();
		float alpha = rho / p.dot(q);

		k_out = k_out + alpha * p;
		r = r - alpha * q;
		rho_1 = rho;

		//TODO: Debug print
//		std::cout << rho << std::endl;
//		cv::namedWindow("r", cv::WINDOW_NORMAL);
//		cv::imshow("r", 0.0001 * r);
//		cv::waitKey(10);

		if (rho < tol) {
			break;
		}
	} // for (int iter = 0;..

	//TODO: Debug print
//	cv::namedWindow("k_out_cg", cv::WINDOW_NORMAL);
//	cv::imshow("k_out_cg", 100*k_out);
//	cv::waitKey(10);
}

void pcg_kernel_core_irls_conv(const cv::Mat& k, const std::vector<cv::Mat>& X,
		const std::vector<cv::Mat>& flipX, int ks, const cv::Mat& weights_l1,
		cv::Mat& out) {
	// This function applies the left hand side of the IRLS system to the
	// kernel x. Uses conv2's.

	// first term: X'*X*k (quadratic term)
	cv::Mat out_l2(k.rows, k.cols, CV_32FC1);
	out_l2 = cv::Scalar(0.0);

    Convolver conv;
	for (unsigned int i = 0; i < X.size(); i++) {

		cv::Mat tmp1(X[i].rows - k.rows, X[i].cols - k.cols, CV_32FC1);

		conv((float*) X[i].data, X[i].cols, X[i].rows, X[i].cols,
				(float*) k.data, k.cols, k.rows, k.cols, (float*) tmp1.data,
				tmp1.cols, tmp1.rows, tmp1.cols);

		cv::Mat tmp3(flipX[i].rows - tmp1.rows, flipX[i].cols - tmp1.cols,
				CV_32FC1);

		conv((float*) flipX[i].data, flipX[i].cols, flipX[i].rows,
				flipX[i].cols, (float*) tmp1.data, tmp1.cols, tmp1.rows,
				tmp1.cols, (float*) tmp3.data, tmp3.cols, tmp3.rows, tmp3.cols);

		/*
		 cv::Mat tmp1 = X[i].clone();
		 tmp1 = cv::Scalar(0);
		 copyImageToKernel((float*) X[i].data, X[i].cols, X[i].rows, X[i].cols,
		 (float*) tmp1.data, tmp1.cols, tmp1.rows, tmp1.cols);
		 cv::dft(tmp1, tmp1);

		 cv::Mat tmp2 = X[i].clone();

		 cv::Mat tmp1Kernel = tmp1.clone();
		 tmp1Kernel = cv::Scalar(0);
		 k.copyTo(
		 tmp1Kernel(
		 cv::Rect(tmp1Kernel.cols / 2 - k.cols / 2,
		 tmp1Kernel.rows / 2 - k.rows / 2, k.cols,
		 k.rows)));
		 cv::dft(tmp1Kernel, tmp1Kernel);

		 cv::mulSpectrums(tmp1, tmp1Kernel, tmp2, cv::DFT_COMPLEX_OUTPUT);

		 cv::mulSpectrums(tmp2, tmp1, tmp1Kernel, cv::DFT_COMPLEX_OUTPUT, true);

		 cv::dft(tmp1Kernel, tmp1Kernel,
		 cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

		 cv::Mat tmp3(out_l2.rows, out_l2.cols, CV_32FC1);
		 tmp1Kernel(
		 cv::Rect(tmp1.cols / 2 - k.cols / 2, tmp1.rows / 2 - k.rows / 2,
		 k.cols, k.rows)).copyTo(tmp3);
		 //		copyImageToKernel((float*) tmp1Kernel.data, tmp1Kernel.cols,
		 //				tmp1Kernel.rows, tmp1Kernel.cols, (float*) tmp3.data, tmp3.cols,
		 //				tmp3.rows, tmp3.cols);

		 */

		//TODO: Debug print
//		cv::namedWindow("tmp1Kernel", cv::WINDOW_NORMAL);
//		cv::imshow("tmp1Kernel", tmp1Kernel);
//		cv::namedWindow("tmp3", cv::WINDOW_NORMAL);
//		cv::imshow("tmp3", tmp3);
//		cv::waitKey(10);
		cv::add(out_l2, tmp3, out_l2);
	}

	//TODO: Debug print
//	cv::namedWindow("out_l2", cv::WINDOW_NORMAL);
//	cv::imshow("out_l2", 0.001 * out_l2);
//	cv::waitKey(10);

	// second term: L1 regularization
	cv::Mat out_l1;
	cv::multiply(weights_l1, k, out_l1);

	cv::add(out_l1, out_l2, out);

	//TODO: Debug print
//	cv::namedWindow("k", cv::WINDOW_NORMAL);
//	cv::imshow("k", 0.01 * k);
//	cv::namedWindow("out", cv::WINDOW_NORMAL);
//	cv::imshow("out", 0.01 * out);
//	cv::waitKey(10);
}

void copyImageToKernel(const float* pImagedata, int imageCols, int imageRows,
		int imagePpln, float* pKerneldata, int kernelCols, int kernelRows,
		int kernelPpln) {

	for (int row = 0; row < kernelRows; row++) {

		int inputRow;
		if (row < kernelRows / 2) {
			inputRow = imageRows + row - kernelRows / 2;
		} else {
			inputRow = row - kernelRows / 2;
		}

		for (int col = 0; col < kernelCols; col++) {

			int inputCol;
			if (col < kernelCols / 2) {
				inputCol = imageCols + col - kernelCols / 2;
			} else {
				inputCol = col - kernelCols / 2;
			}

			pKerneldata[col + row * kernelPpln] = pImagedata[inputCol
					+ inputRow * imagePpln];
		}
	}

	return;
}

void copyKernelToImage(const float* pKerneldata, int kernelCols, int kernelRows,
		int kernelPpln, float* pImagedata, int imageCols, int imageRows,
		int imagePpln) {

	for (int row = 0; row < kernelRows; row++) {

		int inputRow;
		if (row < kernelRows / 2) {
			inputRow = imageRows + row - kernelRows / 2;
		} else {
			inputRow = row - kernelRows / 2;
		}

		for (int col = 0; col < kernelCols; col++) {

			int inputCol;
			if (col < kernelCols / 2) {
				inputCol = imageCols + col - kernelCols / 2;
			} else {
				inputCol = col - kernelCols / 2;
			}

			pImagedata[inputCol + inputRow * imagePpln] = pKerneldata[col
					+ row * kernelPpln];
		}
	}

	return;
}
