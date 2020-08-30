#include "utility.h"
#include "imageRead.h"

string getFileName(string str)
{
	int m = str.find_last_of('\\');
	int n = str.find_last_of('.');

	return str.substr(m + 1, n - m - 1);
}
void getRangeFromTxt(string tempObject, int *xmin, int *xmax, int *ymin, int *ymax, const Mat &img)
{
	int posi[8];
	int k = 0;
	posi[0] = 0;
	for (int i = 0; i != tempObject.size(); ++i)
	{
		if (tempObject[i] == ',')
		{
			posi[++k] = i;
		}
	}

	int xy[8];

	xy[0] = atoi(tempObject.substr(0, posi[1]).c_str());
	for (k = 1; k < 8; k++)
	{
		xy[k] = atoi(tempObject.substr(posi[k] + 1, posi[k + 1] - posi[k] - 1).c_str());
	}

	/*check premeters*/
	{
		xy[0] = xy[0] < 0 ? 0 : xy[0]; xy[0] = xy[0] > img.cols ? img.cols : xy[0];
		xy[1] = xy[1] < 0 ? 0 : xy[1]; xy[1] = xy[1] > img.rows ? img.rows : xy[1];

		xy[4] = xy[4] < 0 ? 0 : xy[4]; xy[4] = xy[4] > img.cols ? img.cols : xy[4];
		xy[5] = xy[5] < 0 ? 0 : xy[5]; xy[5] = xy[5] > img.rows ? img.rows : xy[5];
	}

	*xmin = xy[0];
	*ymin = xy[1];
	*xmax = xy[4];
	*ymax = xy[5];
}

void processData(vector<string> &rects_str, string &str_box)
{
	for (int i = 0; i < rects_str.size();)
	{
		string tempN = rects_str[i++];
		string tempP = rects_str[i++];
		i++;

		if (!tempP.compare("1\r"))
			str_box = tempN;
	}
}

double iou(const Rect &r1,const Rect &r2)
{
	int in_w = min(r1.x+r1.width, r2.x + r2.width) - max(r1.x, r2.x);
	int in_h = min(r1.y+r1.height, r2.y + r2.height) - max(r1.y, r2.y);

	double in_area = (in_w <= 0 || in_h <= 0) ? 0.f : in_w*in_h;
	double un_area = r1.area() + r2.area() - in_area;

	return	in_area / un_area;
}
double overlapArea(const cv::Rect &a, const cv::Rect &b)
{
	int w = std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x);
	int h = std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y);

	return (w < 0 || h < 0) ? 0.f : (float)(w * h);
}

void normSizeByOneSide(Mat &recImage, double length)
{
	double w = recImage.cols;
	double h = recImage.rows;

	if (max(w, h) == length)
		return;

	if (w > h)
	{
		double s = length / w;
		resize(recImage, recImage, Size(), s, s);
	}
	else
	{
		double s = length / h;
		resize(recImage, recImage, Size(), s, s);
	}
}


void rectScale(Rect &box, double r)
{
	int dx = cvRound((r - 1)*box.width / 2.0);
	int dy = cvRound((r - 1)*box.height / 2.0);
	Rect new_box(box);
	new_box.x -= dx;
	new_box.y -= dy;
	new_box.width = cvRound(r*box.width);
	new_box.height = cvRound(r*box.height);
	box = new_box;
}
void generatPosiOneRect_Scale(const Mat &src, const Rect &bbox, vector<Rect> &boxes)
{
	Rect box = bbox;
	double r_max = 1.05;
	double r_min = 0.9;
	double step = 0.1;
	for (double r = r_min; r <= r_max; r += step)
	{
		int dx = cvRound((r - 1)*box.width / 2.0);
		int dy = cvRound((r - 1)*box.height / 2.0);
		Rect new_box(box);
		new_box.x -= dx;
		new_box.y -= dy;
		new_box.width = cvRound(r*box.width);
		new_box.height = cvRound(r*box.height);
		Mat show = src.clone();
		rectangle(show, new_box, Scalar(0, 0, 255));

		boxes.push_back(new_box);
	}
	
}
void generatPosiOneRect_Offset(const Mat &src, const Rect &bbox, vector<Rect> &boxes)
{
	Rect box = bbox;
	/*int x_left = box.x-box.width*0.01;
	int x_right = box.x + box.width*0.01;

	int y_up = box.y - box.height*0.1;
	int y_down = box.y + box.height*0.1;*/

	int x_left = box.x - 2;
	int x_right = box.x + 2;

	int y_up = box.y - 2;
	int y_down = box.y + 2;

	int step = 2;
	for (int x = x_left; x <= x_right; x += step)
	{
		for (int y = y_up; y <= y_down; y += step)
		{
			Rect new_box(box);
			new_box.x = x;
			new_box.y = y;
			if (x<0 || x + new_box.width>src.cols) continue;
			if (y<0 || y + new_box.height>src.rows) continue;

			Mat show = src.clone();
			rectangle(show, new_box, Scalar(0, 0, 255));
			boxes.push_back(new_box);
		}
	}

}
void generatPosOneRect(const Mat &src, const Rect &bbox, vector<Rect> &boxes)
{
	vector<Rect> boxes_scale;
	generatPosiOneRect_Scale(src, bbox, boxes_scale);
	vector<Rect> boxes_offset;
	for (auto r : boxes_scale)
	{
		generatPosiOneRect_Offset(src, r, boxes_offset);
	}
	boxes.insert(boxes.end(),boxes_offset.begin(), boxes_offset.end());
}
void prepareSamples(const Mat &src, const Rect &bbox, const Size &de_window, vector<Rect> &sampels)
{
	if (bbox.width < de_window.width*0.9 || bbox.height < de_window.height*0.9) return;
	Mat watermark(src, bbox);
	Mat wm = watermark.clone();
	resize(wm, wm, Size(de_window.width,de_window.height));
	double de_r = double(de_window.width) / de_window.height;
	double gth_r = double(bbox.width) / bbox.height;

	Rect fix_bbox1(bbox), fix_bbox2(bbox);
	if (gth_r != de_r)
	{
		int new_h = cvRound(bbox.width/de_r);
		fix_bbox1.height = new_h;

		int new_w = cvRound(bbox.height * de_r);
		fix_bbox2.width = new_w;

		Mat show_1 = src.clone();
		Mat show_2 = src.clone();
		rectangle(show_1, fix_bbox1, Scalar(0, 0, 255));
		rectangle(show_2, fix_bbox2, Scalar(0, 0, 255));
	}

	generatPosOneRect(src, fix_bbox1, sampels);
	generatPosOneRect(src, fix_bbox2, sampels);

}
