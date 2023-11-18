/*
 * deconv_opts.cpp
 *
 *  Created on: Jan 9, 2014
 *      Author: vladiant
 */

#include "deconv_opts.h"

BlindDeblurOptions::BlindDeblurOptions() {
	prescale = 1.0;
	min_lambda = 100.0;
	k_reg_wt = 0.0;
	lambda = 0.0;
	gamma_correct = 1.0;
	k_thresh = 0.0;
	kernel_init = 3;
	delta = 0.001;
	x_in_iter = 2;
	x_out_iter = 2;
	xk_iter = 21;
	nb_lambda = 3000;
	nb_alpha = 1.0;
	use_ycbcr = 1;
	kernel_size = 31;
	use_fft = true;
	pcg_tol=1e-4;
	pcg_its = 1;
}


