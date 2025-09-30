// Author: Jacopo Onorati

#include <opencv2/opencv.hpp>

#include "image.h"

using namespace cv;

// Return the final output image
Mat Image::get_pic() {
    return this->_pic;
}

// Set the final output image
void Image::set_pic(Mat& pic) {
    this->_pic = pic;
}

// Return the vector of ROI images.
std::vector<Mat> Image::get_ROI() {
    return this->_roi_image;
}

// Add a new ROI image to the vector .
void Image::set_ROI(Mat& roi) {
    this->_roi_image.push_back(roi);
}

// Return the vector of preprocessed ROI images.
std::vector<Mat> Image::get_preprocessed_ROI() {
    return this->preprocessed_ROI;
}

// Set the vector containing images with preprocessed ROIs
void Image::set_preprocessed_ROI(std::vector<Mat> prepr_roi) {
    this->preprocessed_ROI = prepr_roi;
}
