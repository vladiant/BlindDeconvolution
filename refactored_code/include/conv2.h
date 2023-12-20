#pragma once

void conv2(const float* imageData, int imageWidth, int imageHeight,
		int imagePpln, const float* kernelData, int kernelWidth,
		int kernelHeight, int kernelPpln, float* convolutionData,
		int convolutionWidth, int convolutionHeight, int convolutionPpln);

