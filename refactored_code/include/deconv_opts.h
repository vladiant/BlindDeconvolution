#pragma once

#include <opencv2/core/types.hpp>

/// TODO: Set these as predefined presets
/// Note that the min_lambda parameter should be varied for different images to
/// give better results.
/// fn = 'lyndsey.tif'; kernel_est_win = [335 275 1170 712]; min_lambda = 60;
/// fn = 'pietro.tif'; kernel_est_win = [141 123 601 828]; min_lambda  = 100;
/// fn = 'mukta.jpg'; kernel_size = 27; use_ycbcr = 0; min_lambda = 200;
/// fn = 'fishes.jpg'; kernel_size = 31; (From Cho/Lee et. al. SIGGRAPH Asia 2009)
/// fn = 'Levin09_im08_filt04_blurry.tif'; kernel_size = 31;
class BlindDeblurOptions {
public:

	/// set kernel_est_win to be the window used for estimating the kernel - if
	/// this option is empty, whole image will be used
	cv::Rect kernel_est_win;

	/// set initial downsampling size for really large images
	float prescale;

	/// This is the weight on the likelihood term - it should be decreased for
	/// noisier images; decreasing it usually makes the kernel "fatter";
	/// increasing makes the kernel "thinner".
	float min_lambda;

	/// weight of the likelihood term
	float lambda;

	/// Kernel regularization weight
	float k_reg_wt;

	/// set this to 1 for no gamma correction - default 1.0
	float gamma_correct;

	/// threshold on fine scale kernel elements
	float k_thresh;

	/// kernel initialiazation at coarsest level
	/// 0 = uniform; 1 = vertical bar; 2 = horizontal bar; 3 = tiny 2-pixel
	/// wide kernel at coarsest level
	int kernel_init;

	/// delta step size for ISTA updates; increasing this delta size is not a
	/// good idea since it may cause divergence. On the other hand decreasing
	/// it too much will make convergence much slower.
	float delta;

	/// inner iterations for x estimation
	int x_in_iter;

	/// outer iterations for x estimation
	int x_out_iter;

	/// maximum number of x/k alternations per level; this is a trade-off
	/// between performance and quality.
	int xk_iter;

	/// non-blind settings
	/// Maximal resudual tolerance for Bregman solution
	float nb_lambda;
	/// Maximal iterations for Bregman solution 
	float nb_alpha;
	/// Flag to use YCbCr color space
	int use_ycbcr;

	int kernel_size;

    // TODO: Unused
	bool use_fft;

	/// Maximal resudual tolerance for CG solution
	float pcg_tol;

	/// Maximal iterations for CG solution 
	int pcg_its;

	BlindDeblurOptions();
};

