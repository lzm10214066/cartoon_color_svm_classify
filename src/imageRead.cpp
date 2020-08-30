#include "imageRead.h"

void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int readImageFile(string &imageFile, vector<string> &pathOfImage)
{
	string buf;
	int imageCount = 0;

	ifstream img_list(imageFile);

	if (img_list)
	{
		cout << "file is : " << imageFile << endl;

	}
	else
	{
		cout << "cannot open the imagelist file." << endl;
	}

	while (img_list)
	{
		if (getline(img_list, buf))
		{
			pathOfImage.push_back(buf);
			imageCount++;
		}
	}
	img_list.close();
	cout << imageCount << " things have been read" << endl;

	return imageCount;
}

void rect2image(Mat &image, vector<Rect> recBag, string imagePath,string &folder,int type)
{
	int m = imagePath.find_last_of('\\');
	int n = imagePath.find_last_of('.');

	string image_name = imagePath.substr(m + 1, n - m - 1);

	Rect rec;
	char dir[50];
	
	sprintf(dir, "%s/%s", folder.c_str(), image_name.c_str());
	_mkdir(dir);

	if (type)
	{
		char dir_p[50];
		sprintf(dir_p, "%s/%s", dir, "p");
		_mkdir(dir_p);

		for (int k = 0; k != recBag.size(); ++k)
		{
			rec = recBag[k];
			Mat recImage(image, rec);
			char recToSaved[50];
			sprintf(recToSaved, "%s/%s_%d_rec.png", dir_p, image_name.c_str(), k);
			imwrite(recToSaved, recImage);
		}
	}

	else
	{
		char dir_n[50];
		sprintf(dir_n, "%s/%s", dir, "n");
		_mkdir(dir_n);

		for (int k = 0; k != recBag.size(); ++k)
		{
			rec = recBag[k];
			Mat recImage(image, rec);
			char recToSaved[50];
			sprintf(recToSaved, "%s/%s_%d_rec.png", dir_n, image_name.c_str(), k);
			imwrite(recToSaved, recImage);
		}
	}
}

void rectsSaveAsImage(Mat &image, vector<Rect> recBag, string imagePath, string &folder)
{
	int m = imagePath.find_last_of('\\');
	int n = imagePath.find_last_of('.');

	string image_name = imagePath.substr(m + 1, n - m - 1);

	for (int k = 0; k != recBag.size(); ++k)
	{
		Rect rec = recBag[k];
		if (rec.x<0 || rec.x + rec.width>image.cols || rec.y<0 || rec.y + rec.height>image.rows) continue;
		Mat recImage(image, rec);
		char recToSaved[500];
		sprintf(recToSaved, "%s/%s_%d_rec.png", folder.c_str(), image_name.c_str(), k);
		imwrite(recToSaved, recImage);
	}
	
}