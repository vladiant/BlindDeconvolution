/*
 * conv2.h
 *
 *  Created on: Feb 16, 2015
 *      Author: vantonov
 */

#ifndef INCLUDE_CONV2_H_
#define INCLUDE_CONV2_H_

#include <assert.h>

#include <iostream>

template<typename ImageT, typename KernelT, typename ConvolvedT>
void conv2(const ImageT* imageData, int imageWidth, int imageHeight,
		int imagePpln, const KernelT* kernelData, int kernelWidth,
		int kernelHeight, int kernelPpln, ConvolvedT* convolutionData,
		int convolutionWidth, int convolutionHeight, int convolutionPpln) {

	assert(imageWidth - kernelWidth == convolutionWidth);
	assert(imageHeight - kernelHeight == convolutionHeight);

	int kernelCenterX = kernelWidth / 2;
	int kernelCenterY = kernelHeight / 2;

	for (int row = kernelCenterY;
			row < imageHeight - (kernelHeight - kernelCenterY); row++) {

		for (int col = kernelCenterX;
				col < imageWidth - (kernelWidth - kernelCenterX); col++) {

			ConvolvedT sum = 0;
			for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
				for (int kernelCol = 0; kernelCol < kernelWidth; kernelCol++) {
					sum += imageData[col - kernelCol + kernelCenterX
							+ (row - kernelRow + kernelCenterY) * imagePpln]
							* kernelData[kernelCol + kernelRow * kernelPpln];
				}
			}

			convolutionData[col - kernelCenterX
					+ (row - kernelCenterY) * convolutionPpln] = sum;

		}
	}

	return;
}

template<>
void conv2(const float* imageData, int imageWidth, int imageHeight,
		int imagePpln, const float* kernelData, int kernelWidth,
		int kernelHeight, int kernelPpln, float* convolutionData,
		int convolutionWidth, int convolutionHeight, int convolutionPpln);

#endif /* INCLUDE_CONV2_H_ */
