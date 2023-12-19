//  This package contains the implementation of the blind deconvolution
//  algorithm presented in the paper:
//
//  "Blind Deconvolution using a Normalized Sparsity Measure", Dilip
//  Krishnan, Terence Tay and Rob Fergus.
//
//  Use of this code is free for research purposes only.
//
//  (c) Dilip Krishnan, Rob Fergus. Email: dilip@cs.nyu.edu, May 2011.


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <iostream>

#include "deconv_opts.h"
#include "ms_blind_deconv.h"

constexpr auto kBlurredImageName = "pietro.tif";

int main(int argc, char* argv[]) {

	cv::Mat blurredImage;
	if (argc > 1) {
		std::cout << "Loading " << argv[1] << std::endl;
		blurredImage = cv::imread(argv[1]);
	} else {
		blurredImage = cv::imread(kBlurredImageName);
		std::cout << "Loading " << kBlurredImageName << std::endl;
	}

	if (blurredImage.empty()) {
		std::cerr << "Error loading image." << std::endl;
		return EXIT_FAILURE;
	}

	cv::Mat kernelImage;
	cv::Mat deblurredImage = blurredImage.clone();

	const BlindDeblurOptions opts;

	ms_blind_deconv(blurredImage, opts, kernelImage, deblurredImage);

	cv::namedWindow("Blurred Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Blurred Image", blurredImage);
	cv::namedWindow("Deblurred Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Deblurred Image", deblurredImage);

	cv::imwrite("deblurred.bmp", deblurredImage * 255);

	cv::waitKey(0);

	std::cout << "Done." << std::endl;

	return EXIT_SUCCESS;
}

