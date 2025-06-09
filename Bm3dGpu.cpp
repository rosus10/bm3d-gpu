#include "Bm3dGpu.h"
#include "bm3d.hpp"

#include <opencv2/imgproc.hpp>
#include <memory>

Bm3dGpu::Bm3dGpu(float sigma, bool two_step)
    : impl_(new BM3D()), two_step_(two_step), default_sigma_(sigma) {
    impl_->set_hard_params(19, 8, 16, 2500, 3, 2.7f);
    impl_->set_wien_params(19, 8, 32, 400, 3);
    impl_->set_verbose(false);
}

cv::Mat Bm3dGpu::operator()(const cv::Mat& src, float sigma) {
    CV_Assert(src.channels() == 1);

    float used_sigma = sigma;
    if (sigma < 0.f)
        used_sigma = default_sigma_;

    cv::Mat src8;
    bool is16 = src.depth() == CV_16U;
    if (is16) {
        src.convertTo(src8, CV_8U, 1.0 / 256.0);
    } else {
        src8 = src.clone();
    }

    cv::Mat dst8(src8.size(), CV_8U);
    unsigned int sigma2 = static_cast<unsigned int>(used_sigma * used_sigma);
    impl_->denoise_host_image(src8.data, dst8.data,
                              src8.cols, src8.rows, 1,
                              &sigma2, two_step_);

    if (is16) {
        cv::Mat dst16;
        dst8.convertTo(dst16, CV_16U, 256.0);
        return dst16;
    }
    return dst8;
}

Bm3dGpu::~Bm3dGpu() {
    delete impl_;
}
