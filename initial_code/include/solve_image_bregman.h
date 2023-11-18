/*
 * solve_image_bregman.h
 *
 *  Created on: Feb 17, 2015
 *      Author: vantonov
 */

#ifndef INCLUDE_SOLVE_IMAGE_BREGMAN_H_
#define INCLUDE_SOLVE_IMAGE_BREGMAN_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void solve_image_bregman(const cv::Mat& image, float beta, float alpha,
		cv::Mat& wx);

#endif /* INCLUDE_SOLVE_IMAGE_BREGMAN_H_ */
