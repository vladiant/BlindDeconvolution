/*
 * ss_blind_deconv.h
 *
 *  Created on: Jan 12, 2014
 *      Author: vladiant
 */

#ifndef SS_BLIND_DECONV_H_
#define SS_BLIND_DECONV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "deconv_opts.h"

int ss_blind_deconv(cv::Mat& y, cv::Mat& x, cv::Mat& k, float lambda,
                    const BlindDeblurOptions& opts,
                    BlindDeblurContext& aContext);

#endif /* SS_BLIND_DECONV_H_ */
