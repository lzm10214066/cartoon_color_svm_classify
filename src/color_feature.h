#pragma once
#include <iostream>

#include "opencv2\opencv.hpp"

#include <set>


#include "Color_Coherence_Vectors\CCV.hpp"

using namespace std;
using namespace cv;

class Color_Feature_Pro
{
	int numColors;
    double coherenceThreshold_r;
	int obj_len;

	void ImgResize(const cv::Mat& img, cv::Mat& resized_img, const int obj_len);
public:
	Color_Feature_Pro(int _numColors = 32, double _coherenceThreshold_r = 0.1,int _object_len=64) :numColors(_numColors), 
		coherenceThreshold_r(_coherenceThreshold_r), obj_len(64)
	{}
	void calculateCCV_feature(const Mat &img, vector<double> &hist);

	void calculateCCV_hist(const Mat &img, vector<double> &hist);
	void calculateCCV_hist_chop(const Mat &img, vector<double> &hist);

	void calculateCCV_Qhist(const Mat &img, vector<double> &hist);
	void calculateCCV_Qhist_chop(const Mat &img, vector<double> &hist);
};