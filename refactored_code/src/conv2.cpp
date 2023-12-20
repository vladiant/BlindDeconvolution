#include "conv2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pcg_kernel_irls_conv.h"

void conv2(const float* imageData, int imageWidth, int imageHeight,
		int imagePpln, const float* kernelData, int kernelWidth,
		int kernelHeight, int kernelPpln, float* convolutionData,
		int convolutionWidth, int convolutionHeight, int convolutionPpln) {

	cv::Mat image(imageHeight, imageWidth, CV_32FC1, (void*) imageData);
	cv::Mat kernel(kernelHeight, kernelWidth, CV_32FC1, (void*) kernelData);
	cv::Mat convolution(convolutionHeight, convolutionWidth,
	CV_32FC1, convolutionData);

	cv::Mat imageFFT = image.clone();
	cv::dft(image, imageFFT);

	cv::Mat kernelFFT = image.clone();
	kernelFFT = cv::Scalar(0);
	copyKernelToImage((float*) kernel.data, kernel.cols, kernel.rows,
			kernel.cols, (float*) kernelFFT.data, kernelFFT.cols,
			kernelFFT.rows, kernelFFT.cols);

	cv::dft(kernelFFT, kernelFFT);

	cv::mulSpectrums(imageFFT, kernelFFT, imageFFT, cv::DFT_COMPLEX_OUTPUT);

	cv::dft(imageFFT, imageFFT,
			cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

	imageFFT(
			cv::Rect(kernelWidth / 2, kernelHeight / 2, convolutionWidth,
					convolutionHeight)).copyTo(convolution);

//	cv::namedWindow("imageFFT", cv::WINDOW_NORMAL);
//	cv::imshow("imageFFT", imageFFT);
//	cv::namedWindow("convolution", cv::WINDOW_NORMAL);
//	cv::imshow("convolution", convolution);
//	cv::waitKey(0);
}
