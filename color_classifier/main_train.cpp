#include <stdio.h>
#include <stdlib.h>
#include <io.h>

#include <iostream>
#include <fstream>

#include "opencv2\opencv.hpp"

#include "../src/imageRead.h"
#include "../src/utility.h"
#include "../src/color_feature.h"

using namespace cv;
using namespace std;

int main(void)
{
	//////////////////////////////////////*read path.txt*///////////////////////////
	string pos_imageFile = "pos_path.txt";
	vector<string> pos_image_path;
	int posCount = readImageFile(pos_imageFile, pos_image_path);

	string neg_imageFile = "neg_path.txt";
	vector<string> neg_image_path;
	int negCount = readImageFile(neg_imageFile, neg_image_path);

	//string image_folder = "C:/Users/lzm/Desktop/watermark_prepare/watermark_detection/neg_images";
	//getFiles(image_folder, image_path);
	/////////////////////////////////////////////////////////////////////////////////
	string featureFile = "color_coherence_vectors_Qhist_chop.xml";
	FileStorage fs_feature(featureFile, FileStorage::WRITE);
	int pos_use = min(5000, posCount);
	int neg_use = min(2000, negCount);

	int imageNum = pos_use + neg_use;
	int featureDim = 792;
	fs_feature << "imageNum" << imageNum;
	fs_feature << "featureDim" << featureDim;
	fs_feature << "lable_feature" << "[";

	Color_Feature_Pro ccv(16,0.1,64);
	/*positive*/
	cout << "\n pos:  " << endl;
	int pos_count = 0;
	for (int i = 0; i != pos_use; ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		string temp = pos_image_path[i];
		Mat imgRGB = imread(temp);
		if (!imgRGB.data) continue;
		pos_count++;

		double t = (double)getTickCount();
		vector<double> feature;
	/*	Mat imgHSV;
		cvtColor(imgRGB, imgHSV, CV_BGR2HSV);*/
		//ccv.calculateCCV_feature(imgRGB, feature);

		//ccv.calculateCCV_hist(imgRGB, feature);
		//ccv.calculateCCV_hist_chop(imgRGB, feature);

		//ccv.calculateCCV_Qhist(imgRGB, feature);
		ccv.calculateCCV_Qhist_chop(imgRGB, feature);

		t = ((double)getTickCount() - t) / getTickFrequency();
		if ((i + 1) % 100 == 0)  cout << "Times passed in ms: " << t * 1000 << endl;
		fs_feature << 1 << feature;
	}
	CV_Assert(pos_count == pos_use);
	/*negative*/
	cout << "\n neg: " << endl;
	int neg_count = 0;
	for (int i = 0; neg_count != neg_use; ++i)
	{
		if ((i + 1) % 100 == 0) cout << i << endl;
		Mat imgRGB = imread(neg_image_path[i]);
		if (!imgRGB.data) continue;
		neg_count++;

		vector<double> feature;
	/*	Mat imgHSV;
		cvtColor(imgRGB, imgHSV, CV_BGR2HSV);*/
		//ccv.calculateCCV_feature(imgRGB, feature);

		//ccv.calculateCCV_hist(imgRGB, feature);
		//ccv.calculateCCV_hist_chop(imgRGB, feature);

		//ccv.calculateCCV_Qhist(imgRGB, feature);
		ccv.calculateCCV_Qhist_chop(imgRGB, feature);

		fs_feature << -1 << feature;
	}
	fs_feature << "]";
	cout << featureFile << "   saved" << endl;
	return EXIT_SUCCESS;
}
