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

double color_classify_one_svm(Mat &img, CvSVM &svm);

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
	CvSVM svm;
	string svmFile = "color_coherence_vectors_hist_chop-SVM-r.xml";
	svm.load(svmFile.c_str());

	int pos_use = min(5000, posCount);
	int neg_use = min(8000, negCount);

	/*positive*/
	cout << "\n pos:  " << endl;
	int pos_count = 0;
	for (int i = 0; i < pos_use; ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		string temp = pos_image_path[i];
		Mat imgRGB = imread(temp);
		if (!imgRGB.data) continue;
		
		double t = (double)getTickCount();
		double s = -color_classify_one_svm(imgRGB, svm);
		if (s>=1) 
			pos_count++;
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << s << endl;
		if ((i + 1) % 100 == 0)
		{
			cout << "Times passed in ms: " << t * 1000 << endl;
			waitKey(0);
		}
		
	}
	cout << pos_count <<"//"<<pos_use<< endl;
	/*negative*/
	cout << "\n neg: " << endl;
	int neg_count = 0;
	for (int i = 2000; i != neg_use && i!=negCount; ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		Mat imgRGB = imread(neg_image_path[i]);
		if (!imgRGB.data) continue;

		double s = -color_classify_one_svm(imgRGB, svm);
		if (s>0) neg_count++;
	}
	cout << neg_count << "//" << neg_use << endl;
	return EXIT_SUCCESS;
}



double color_classify_one_svm(Mat &img, CvSVM &svm)
{
	Color_Feature_Pro ccv(16, 0.1, 64);
	vector<double> hist;
	/*Mat imgHSV;
	cvtColor(img, imgHSV, CV_BGR2HSV);*/
	//ccv.calculateCCV_feature(img, hist);

	//ccv.calculateCCV_hist(img, hist);
	ccv.calculateCCV_hist_chop(img, hist);

	//ccv.calculateCCV_Qhist(img, hist);
	//ccv.calculateCCV_Qhist_chop(img, hist);

	Mat hist_mat(1, hist.size(), CV_32FC1);
	for (int i = 0; i < hist.size(); ++i)
	{
		hist_mat.at<float>(0, i) = hist[i];
	}
	return(svm.predict(hist_mat, true));
}