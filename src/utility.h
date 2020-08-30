#pragma once
#include <iostream>

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\ml\ml.hpp"
#include "opencv2\features2d\features2d.hpp"
#include "opencv2\objdetect\objdetect.hpp"

#include <set>

using namespace std;
using namespace cv;

string getFileName(string str);
void getRangeFromTxt(string tempObject, int *xmin, int *xmax, int *ymin, int *ymax,const Mat &img);
void processData(vector<string> &rects_str, string &str_box);
double iou(const Rect &r1, const Rect &r2);
double overlapArea(const cv::Rect &a, const cv::Rect &b);
void normSizeByOneSide(Mat &recImage, double length);
void prepareSamples(const Mat &src, const Rect &bbox, const Size &de_window, vector<Rect> &sampels);
void rectScale(Rect &box, double r);




