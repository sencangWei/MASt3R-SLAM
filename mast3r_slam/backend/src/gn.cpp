#include "gn.h"

std::vector<torch::Tensor> gauss_newton_points(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_points_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_point, C_thresh, Q_thresh, max_iter, delta_thresh);
}

std::vector<torch::Tensor> gauss_newton_rays(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_rays_cuda(Twc, Xs, Cs, ii, jj, idx_ii2jj, valid_match, Q,
      sigma_ray, sigma_dist, C_thresh, Q_thresh, max_iter, delta_thresh);
}

std::vector<torch::Tensor> gauss_newton_calib(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh) {

  CHECK_CONTIGUOUS(Twc);
  CHECK_CONTIGUOUS(Xs);
  CHECK_CONTIGUOUS(Cs);
  CHECK_CONTIGUOUS(K);
  CHECK_CONTIGUOUS(ii);
  CHECK_CONTIGUOUS(jj);
  CHECK_CONTIGUOUS(idx_ii2jj);
  CHECK_CONTIGUOUS(valid_match);
  CHECK_CONTIGUOUS(Q);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return gauss_newton_calib_cuda(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh, max_iter, delta_thresh,
      torch::empty({0}, Twc.options()), torch::empty({0}, Twc.options().dtype(torch::kBool)), 0.0f, 0.0f);
}

std::vector<torch::Tensor> gauss_newton_calib_metric(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs, torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, torch::Tensor idx_ii2jj,
  torch::Tensor valid_match, torch::Tensor Q,
  const int height, const int width, const int pixel_border, const float z_eps,
  const float sigma_pixel, const float sigma_depth, const float C_thresh,
  const float Q_thresh, const int max_iter, const float delta_thresh,
  torch::Tensor metric_targets, torch::Tensor metric_valid,
  const float metric_position_sigma, const float metric_log_scale_sigma) {
  CHECK_CONTIGUOUS(metric_targets);
  CHECK_CONTIGUOUS(metric_valid);
  TORCH_CHECK(metric_targets.dim() == 2 && metric_targets.size(1) == 4 && metric_targets.size(0) == Twc.size(0),
      "metric_targets must be [num_poses, 4]");
  TORCH_CHECK(metric_valid.dim() == 1 && metric_valid.size(0) == Twc.size(0),
      "metric_valid must have one entry per pose");
  TORCH_CHECK(metric_position_sigma > 0 && metric_log_scale_sigma > 0,
      "metric prior sigmas must be positive");
  return gauss_newton_calib_cuda(Twc, Xs, Cs, K, ii, jj, idx_ii2jj, valid_match, Q,
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh,
      max_iter, delta_thresh, metric_targets, metric_valid,
      metric_position_sigma, metric_log_scale_sigma);
}

std::vector<torch::Tensor> iter_proj(
  torch::Tensor rays_img_with_grad, 
  torch::Tensor pts_3d_norm, 
  torch::Tensor p_init,
  const int max_iter,
  const float lambda_init,
  const float cost_thresh) {

  CHECK_CONTIGUOUS(rays_img_with_grad);
  CHECK_CONTIGUOUS(pts_3d_norm);
  CHECK_CONTIGUOUS(p_init);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return iter_proj_cuda(rays_img_with_grad, pts_3d_norm, p_init, 
      max_iter, lambda_init, cost_thresh);
}

std::vector<torch::Tensor> refine_matches(
  torch::Tensor D11, 
  torch::Tensor D21,
  torch::Tensor p1, 
  const int window_size,
  const int dilation_max) {

  CHECK_CONTIGUOUS(D11);
  CHECK_CONTIGUOUS(D21);
  CHECK_CONTIGUOUS(p1);

  // const at::cuda::OptionalCUDAGuard device_guard(device_of(x1));
  return refine_matches_cuda(D11, D21, p1, window_size, dilation_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gauss_newton_points", &gauss_newton_points, "gauss_newton point adjustment");
  m.def("gauss_newton_rays", &gauss_newton_rays, "gauss_newton ray adjustment");
  m.def("gauss_newton_calib", &gauss_newton_calib, "gauss_newton calib adjustment");
  m.def("gauss_newton_calib_metric", &gauss_newton_calib_metric, "calibrated graph adjustment with metric keyframe priors");

  m.def("iter_proj", &iter_proj, "iterative projection with generic camera");
  m.def("refine_matches", &refine_matches, "refine match in local neighborhood");
}
