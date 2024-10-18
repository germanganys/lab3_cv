#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cstdlib>

void manual_binarization(const cv::Mat& src, cv::Mat& dst, uint8_t threshold) {
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            uint8_t pixel_value = src.at<uint8_t>(i, j);
            dst.at<uint8_t>(i, j) = (pixel_value >= threshold) ? 255 : 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./manual_binarization <threshold> <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat manual_binary_image = cv::Mat::zeros(image.size(), image.type());

    uint8_t threshold_value = atoi(argv[1]);

    // Измеряем время выполнения ручной бинаризации
    auto start = std::chrono::high_resolution_clock::now();
    manual_binarization(image, manual_binary_image, threshold_value);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> manual_duration = end - start;
    std::cout << "Manual binarization time: " << manual_duration.count() << " seconds." << std::endl;

    cv::imwrite("manual_binary_output.jpg", manual_binary_image);

    return 0;
}