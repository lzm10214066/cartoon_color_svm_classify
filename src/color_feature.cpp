#include "color_feature.h"


void Color_Feature_Pro::ImgResize(const cv::Mat& img, cv::Mat& resized_img, const int obj_len)
{
	int min_len = min(img.rows, img.cols);
	if (min_len < obj_len) return;

	double s = double(obj_len) / min_len;
	resize(img, resized_img, cv::Size(int(s * img.cols), int(s * img.rows)));
}

void Color_Feature_Pro::calculateCCV_feature(const Mat &img, vector<double> &hist)
{
	Mat imgPro;
	ImgResize(img, imgPro, obj_len);
	int coherenceThreshold = imgPro.rows*imgPro.cols*coherenceThreshold_r;
	lssr::CCV* ccv1 = new lssr::CCV(imgPro, numColors, coherenceThreshold);
	ccv1->calculateImgCCV(imgPro);
	std::map< uchar, std::pair<ulong, ulong> >::iterator ccvit;

	if (!hist.empty()) hist.clear();
	//r
	for (ccvit = ccv1->m_CCV_r.begin(); ccvit != ccv1->m_CCV_r.end(); ccvit++)
	{
		//|alpha| + |beta|
		hist.push_back(ccvit->second.first / (1.0f * ccv1->m_numPix));
		hist.push_back(ccvit->second.second / (1.0f * ccv1->m_numPix));
	}
	//g
	for (ccvit = ccv1->m_CCV_g.begin(); ccvit != ccv1->m_CCV_g.end(); ccvit++)
	{
		//|alpha| + |beta|
		hist.push_back(ccvit->second.first / (1.0f * ccv1->m_numPix));
		hist.push_back(ccvit->second.second / (1.0f * ccv1->m_numPix));
	}
	//b
	for (ccvit = ccv1->m_CCV_b.begin(); ccvit != ccv1->m_CCV_b.end(); ccvit++)
	{
		//|alpha| + |beta|
		hist.push_back(ccvit->second.first / (1.0f * ccv1->m_numPix));
		hist.push_back(ccvit->second.second / (1.0f * ccv1->m_numPix));
	}
}

void Color_Feature_Pro::calculateCCV_hist(const Mat &img, vector<double> &hist)
{
	Mat imgPro;
	ImgResize(img, imgPro, obj_len);
	lssr::CCV* ccv1 = new lssr::CCV(imgPro, numColors);
	ccv1->calculateImgCCV_hist(imgPro);
	std::map< uchar, std::vector<double> >::iterator ccvit;

	if (!hist.empty()) hist.clear();
	//r
	for (ccvit = ccv1->m_CCV_r_hist.begin(); ccvit != ccv1->m_CCV_r_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist.push_back(c / (1.0f * ccv1->m_numPix));
		}
	}
	//g

	for (ccvit = ccv1->m_CCV_g_hist.begin(); ccvit != ccv1->m_CCV_g_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist.push_back(c / (1.0f * ccv1->m_numPix));
		}
	}
	//b
	for (ccvit = ccv1->m_CCV_b_hist.begin(); ccvit != ccv1->m_CCV_b_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist.push_back(c / (1.0f * ccv1->m_numPix));
		}
	}
}

void Color_Feature_Pro::calculateCCV_hist_chop(const Mat &img, vector<double> &hist)
{
	Mat imgPro;
	ImgResize(img, imgPro, obj_len);
	lssr::CCV* ccv1 = new lssr::CCV(imgPro, numColors);
	ccv1->calculateImgCCV_hist(imgPro);
	std::map< uchar, std::vector<double> >::iterator ccvit;

	if (!hist.empty()) hist.clear();
	int max_p = 0;
	int max_c = 0;
	int p = 0;
	vector<double> hist_r;
	//r
	for (ccvit = ccv1->m_CCV_r_hist.begin(); ccvit != ccv1->m_CCV_r_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist_r.push_back(c / (1.0f * ccv1->m_numPix));
			if (c > max_c)
			{
				max_c = c;
				max_p = p;
			}
			++p;
		}
	}
	hist_r[max_p] = 0;
	normalize(hist_r,hist_r, 1, 0, NORM_L1);
	//g
	max_p = 0;
	max_c = 0;
	p = 0;
	vector<double> hist_g;
	for (ccvit = ccv1->m_CCV_g_hist.begin(); ccvit != ccv1->m_CCV_g_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist_g.push_back(c / (1.0f * ccv1->m_numPix));
			if (c > max_c)
			{
				max_c = c;
				max_p = p;
			}
			++p;
		}
	}
	hist_g[max_p] = 0;
	normalize(hist_g, hist_g, 1, 0, NORM_L1);
	//b
	max_p = 0;
	max_c = 0;
	p = 0;
	vector<double> hist_b;
	for (ccvit = ccv1->m_CCV_b_hist.begin(); ccvit != ccv1->m_CCV_b_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist_b.push_back(c / (1.0f * ccv1->m_numPix));
			if (c > max_c)
			{
				max_c = c;
				max_p = p;
			}
			++p;
		}
	}
	hist_b[max_p] = 0;
	normalize(hist_b, hist_b, 1, 0, NORM_L1);

	hist.insert(hist.end(), hist_r.begin(), hist_r.end());
	hist.insert(hist.end(), hist_g.begin(), hist_g.end());
	hist.insert(hist.end(), hist_b.begin(), hist_b.end());

}

void Color_Feature_Pro::calculateCCV_Qhist(const Mat &img, vector<double> &hist)
{
	Mat imgPro;
	ImgResize(img, imgPro, obj_len);
	lssr::CCV* ccv1 = new lssr::CCV(imgPro, 72);
	ccv1->calculateImgCCV_Qhist(imgPro);
	std::map< uchar, std::vector<double> >::iterator ccvit;

	if (!hist.empty()) hist.clear();
	//r
	for (ccvit = ccv1->m_CCV_q_hist.begin(); ccvit != ccv1->m_CCV_q_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist.push_back(c / (1.0f * ccv1->m_numPix));
		}
	}
}

void Color_Feature_Pro::calculateCCV_Qhist_chop(const Mat &img, vector<double> &hist)
{
	Mat imgPro;
	ImgResize(img, imgPro, obj_len);
	lssr::CCV* ccv1 = new lssr::CCV(imgPro, 72);
	ccv1->calculateImgCCV_Qhist(imgPro);
	std::map< uchar, std::vector<double> >::iterator ccvit;

	if (!hist.empty()) hist.clear();
	//r
	int max_p = 0;
	int max_c = 0;
	int p = 0;
	for (ccvit = ccv1->m_CCV_q_hist.begin(); ccvit != ccv1->m_CCV_q_hist.end(); ccvit++)
	{
		for (auto c : ccvit->second)
		{
			hist.push_back(c / (1.0f * ccv1->m_numPix));
			if (c > max_c)
			{
				max_c = c;
				max_p = p;
			}
			++p;
		}
	}
	hist[max_p] = 0;
}