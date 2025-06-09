#include <opencv2/imgcodecs.hpp>
#include "Bm3dGpu.h"

int main() {
    cv::Mat noisy = cv::imread("frame.tiff", cv::IMREAD_UNCHANGED);
    Bm3dGpu denoiser(15.0f);
    cv::Mat clean = denoiser(noisy);
    cv::imwrite("clean.tiff", clean);
    return 0;
}
