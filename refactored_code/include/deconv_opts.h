/*
 * deconv_opts.h
 *
 *  Created on: Jan 9, 2014
 *      Author: vladiant
 */

#ifndef BLINDDEBLUROPTIONS_H_
#define BLINDDEBLUROPTIONS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class BlindDeblurOptions {
public:

	// set kernel_est_win to be the window used for estimating the kernel - if
	// this option is empty, whole image will be used
	cv::Rect kernel_est_win;

	// set initial downsampling size for really large images
	float prescale;

	// This is the weight on the likelihood term - it should be decreased for
	// noisier images; decreasing it usually makes the kernel "fatter";
	// increasing makes the kernel "thinner".
	float min_lambda;

	//TODO: Document
	// lambda .
	float lambda;

	// Kernel regularization weight
	float k_reg_wt;

	// set this to 1 for no gamma correction - default 1.0
	float gamma_correct;

	// threshold on fine scale kernel elements
	float k_thresh;

	// kernel initialiazation at coarsest level
	// 0 = uniform; 1 = vertical bar; 2 = horizontal bar; 3 = tiny 2-pixel
	// wide kernel at coarsest level
	int kernel_init;

	// delta step size for ISTA updates; increasing this delta size is not a
	// good idea since it may cause divergence. On the other hand decreasing
	// it too much will make convergence much slower.
	float delta;

	// inner iterations for x estimation
	int x_in_iter;

	// outer iterations for x estimation
	int x_out_iter;

	// maximum number of x/k alternations per level; this is a trade-off
	// between performance and quality.
	int xk_iter;

	// non-blind settings
	float nb_lambda;
	float nb_alpha;
	int use_ycbcr;

	int kernel_size;

	bool use_fft;

	//TODO: Document
	float pcg_tol;

	//TODO: Document
	int pcg_its;

	// Default constructor
	BlindDeblurOptions();
};

#endif /* BLINDDEBLUROPTIONS_H_ */
