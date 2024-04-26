#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

string strPath;
string strSavefilePath;
string strSavefilePath_Img;

Mat img;
Mat bin;
Mat dst;
Mat ResultImg;

Mat HSV_img;

int threshold_low = 0;
int threshold_high = 8;

Mat labels, stats, centroids;
vector<vector<Point>> contours;

vector<Rect> BoundingRect;

bool compare(const Rect &a, const Rect &b)
{
	return a.area() > b.area();
}

void getResult()
{
	Scalar lower_red1(0, 100, 100);
	Scalar upper_red1(10, 255, 255);
	Scalar lower_red2(160, 100, 100);
	Scalar upper_red2(180, 255, 255);

	// 녹색 차량을 검출하기 위한 HSV 범위 설정
	Scalar lower_green(40, 10, 10);
	Scalar upper_green(80, 255, 255);

	// 파란색 차량을 검출하기 위한 HSV 범위 설정
	Scalar lower_blue(100, 10, 10);
	Scalar upper_blue(140, 255, 255);

	// 노란색 차량을 검출하기 위한 HSV 범위 설정
	Scalar lower_yellow(20, 100, 100);
	Scalar upper_yellow(30, 255, 255);

	//검정색 차량을 검출하기 위한 HSV 범위 설정
	Scalar lower_black(0, 0, 0);
	Scalar upper_black(255, 255, 30);

	//White
	Scalar lower_white(0, 0, 200);
	Scalar upper_white(180, 30, 255);

	//purple
	Scalar lower_purple(125, 50, 50);
	Scalar upper_purple(160, 255, 255);

	Mat red_mask1, red_mask2, green_mask, blue_mask, yellow_mask, black_mask, white_mask, purple_mask;
	inRange(HSV_img, lower_red1, upper_red1, red_mask1);
	inRange(HSV_img, lower_red2, upper_red2, red_mask2);
	inRange(HSV_img, lower_green, upper_green, green_mask);
	inRange(HSV_img, lower_blue, upper_blue, blue_mask);
	inRange(HSV_img, lower_yellow, upper_yellow, yellow_mask);
	inRange(HSV_img, lower_black, upper_black, black_mask);
	inRange(HSV_img, lower_white, upper_white, white_mask);
	inRange(HSV_img, lower_purple, upper_purple, purple_mask);

	Mat red_result1, red_result2, green_result, blue_result, yellow_result, black_result, white_result, purple_result;
	//bitwise_or(red_mask1, red_mask2, red_result);
	bitwise_and(img, img, red_result1, red_mask1);
	bitwise_and(img, img, red_result2, red_mask2);
	bitwise_and(img, img, green_result, green_mask);
	bitwise_and(img, img, blue_result, blue_mask);
	bitwise_and(img, img, yellow_result, yellow_mask);
	bitwise_and(img, img, black_result, black_mask);
	bitwise_and(img, img, white_result, white_mask);
	bitwise_and(img, img, purple_result, purple_mask);

	Mat testimg;
	testimg = red_result1 + red_result2;
	testimg = testimg + green_result;
	testimg = testimg + blue_result;
	testimg = testimg + yellow_result;
	testimg = testimg + black_result;
	testimg = testimg + white_result;
	testimg = testimg + purple_result;

	cvtColor(testimg, testimg, COLOR_HSV2BGR);
	cvtColor(testimg, testimg, COLOR_BGR2GRAY);

	inRange(testimg, threshold_low, threshold_high, bin);

	//잡음 제거
	medianBlur(bin, bin, 3);

	//배경부분 잡음 제거
	morphologyEx(bin, bin, MORPH_CLOSE, Mat(), Point(-1, -1), 2);
	morphologyEx(bin, bin, MORPH_ERODE, Mat(), Point(-1, -1), 2);

	bitwise_not(bin, bin);

	imshow("bin", bin);

	dst = bin.clone();

	//Labeling
	int n = connectedComponentsWithStats(dst, labels, stats, centroids);
	//find Contour
	findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	ResultImg = img.clone();

	BoundingRect.clear();

	for (int i = 0; i < (int)contours.size(); i++)
	{
		Rect mr = boundingRect(contours[i]);

		BoundingRect.push_back(mr);
	}

	//sort
	//찾은 Rect의 Area 오름차순 정렬
	sort(BoundingRect.begin(), BoundingRect.end(), ::compare);

	ofstream writeFile(strSavefilePath.data());

	double dRatio = 0.0;

	int nCount = 0;
	for (int i = 0; i < BoundingRect.size(); i++)
	{
		//Rect의 width, height 검사
		if (BoundingRect[i].width > 0 && BoundingRect[i].height > 0 && BoundingRect[i].width < 500 && BoundingRect[i].height < 180)
		{
			//가로세로 비율 : 세로보다 가로로 긴 Rect만 출력
			dRatio = (double)BoundingRect[i].height / (double)BoundingRect[i].width;

			if (dRatio < 0.7)
			{
				if (nCount < 5)
				{
					rectangle(ResultImg, BoundingRect[i], Scalar(0, 255, 255), 1);

					if (writeFile.is_open())
					{
						writeFile << i << "\t" << BoundingRect[i].x << "\t" << BoundingRect[i].y << "\t" << BoundingRect[i].width << "\t" << BoundingRect[i].height << "\n";
					}
					nCount++;
				}
			}
		}
	}
	writeFile.close();
	imwrite(strSavefilePath_Img, ResultImg);
	imshow("img", ResultImg);
}

vector<string> split(string str, char Delimiter)
{
	istringstream iss(str);             // istringstream에 str을 담는다.
	string buffer;                      // 구분자를 기준으로 절삭된 문자열이 담겨지는 버퍼

	vector<string> result;

	// istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
	while (getline(iss, buffer, Delimiter))
	{
		result.push_back(buffer);               // 절삭된 문자열을 vector에 저장
	}

	return result;
}

int main()
{
	//strPath = "D:\\999.Image\\PJT_Image\\test1.png";
	//strPath = "D:\\999.Image\\PJT_Image\\test2.png";
	//strPath = "D:\\999.Image\\PJT_Image\\test3.png";
	//strPath = "D:\\999.Image\\PJT_Image\\test4.png";
	//strPath = "D:\\999.Image\\PJT_Image\\test5.png";
	strPath = "D:\\999.Image\\PJT_Image\\test6.png";
	img = imread(strPath, IMREAD_COLOR);

	vector<string> result_Temp = split(strPath, '\\');
	int nResultpos = result_Temp.size() - 1;

	vector<string> result_Filename = split(result_Temp[nResultpos], '.');

	strSavefilePath = "D:\\999.Image\\PJT_Image\\" + result_Filename[0] + "_Result.txt";
	strSavefilePath_Img = "D:\\999.Image\\PJT_Image\\" + result_Filename[0] + "_Result.png";
	 
	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}
	cvtColor(img, HSV_img, COLOR_BGR2HSV);

	getResult();
	waitKey();

	return 0;
}