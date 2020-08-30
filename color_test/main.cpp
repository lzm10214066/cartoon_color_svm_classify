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

string getFileName(string str)
{
	int m = str.find_last_of('\\');
	int n = str.find_last_of('.');
	if (n <= m) str += ".png";

	return str.substr(m + 1);
}

double color_classify_one_svm(Mat &img, CvSVM &svm);

int main(void)
{
	//////////////////////////////////////*read path.txt*///////////////////////////
	string pos_imageFile = "path.txt";
	vector<string> image_path;
	int count = readImageFile(pos_imageFile, image_path);

	//string image_folder = "C:/Users/lzm/Desktop/watermark_prepare/watermark_detection/neg_images";
	//getFiles(image_folder, image_path);
	/////////////////////////////////////////////////////////////////////////////////
	CvSVM svm;
	string svmFile = "color_coherence_vectors_hist_chop-SVM-r.xml";
	svm.load(svmFile.c_str());

	int use = min(8000, count);

	for (int i = 0; i < use; ++i)
	{
		if ((i + 1) % 100 == 0) cout << i+1 << endl;
		string temp = image_path[i];
		Mat imgRGB = imread(temp);
		if (!imgRGB.data) continue;
		string fileName = getFileName(temp);

		double t = (double)getTickCount();
		double s = -color_classify_one_svm(imgRGB, svm);
		if (s >= 1)
		{
			string folder = "pos";
			string toSave = folder + "/" + fileName;
			imwrite(toSave, imgRGB);
		}
		else
		{
			string folder = "neg";
			string toSave = folder + "/" + fileName;
			//imwrite(toSave, imgRGB);
		}
	
		t = ((double)getTickCount() - t) / getTickFrequency();
		cout << s << endl;
		if ((i + 1) % 100 == 0)
		{
			cout << "Times passed in ms: " << t * 1000 << endl;
			waitKey(0);
		}
		
	}
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