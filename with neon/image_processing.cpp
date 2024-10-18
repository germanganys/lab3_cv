#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <iostream>
#include <chrono>
#include <cstdlib>

void neon_binarization(const cv::Mat& src, cv::Mat& dst, uint8_t threshold) {
    int total_pixels = src.rows * src.cols;
    const uint8_t* src_data = src.data;
    uint8_t* dst_data = dst.data;

    int i = 0;

    // Используем Neon для обработки блоков по 16 байт за раз (uint8x16_t)
    uint8x16_t threshold_vec = vdupq_n_u8(threshold); // Загружаем порог в вектор Neon

    for (; i <= total_pixels - 16; i += 16) {
        // Загружаем 16 пикселей
        uint8x16_t pixels = vld1q_u8(src_data + i);

        // Сравниваем каждый пиксель с порогом
        uint8x16_t result = vcgeq_u8(pixels, threshold_vec);

        // Если больше или равно порогу, устанавливаем 255, иначе 0
        uint8x16_t binary_pixels = vandq_u8(result, vdupq_n_u8(255));

        // Записываем результат в выходное изображение
        vst1q_u8(dst_data + i, binary_pixels);
    }

    // Обрабатываем оставшиеся пиксели, если их меньше 16
    for (; i < total_pixels; ++i) {
        dst_data[i] = src_data[i] >= threshold ? 255 : 0;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./neon_binarization <threshold> <image_path>" << std::endl;
        return -1;
    }

    cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    cv::Mat neon_binary_image = cv::Mat::zeros(image.size(), image.type());

    uint8_t threshold_value = atoi(argv[1]);

    // Измеряем время выполнения бинаризации с использованием Neon
    auto start = std::chrono::high_resolution_clock::now();
    neon_binarization(image, neon_binary_image, threshold_value);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> neon_duration = end - start;
    std::cout << "Neon binarization time: " << neon_duration.count() << " seconds." << std::endl;

    cv::imwrite("neon_binary_output.jpg", neon_binary_image);

    return 0;
}