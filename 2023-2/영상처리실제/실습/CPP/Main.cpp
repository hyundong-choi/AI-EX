#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat src, src_gray, dst;
int threshold_value = 0;
int threshold_type = 0;

Mat img_p26;
int nDrawing_p26 = false;

Mat img_p30;
string title = "트랙바이벤트";

Mat img_p33;
int red, green, blue;
int nDrawing_p33 = false;
void on_trackbar(int, void*) {}

Mat img_hw1;
int nDrawing_hw1 = false;
void on_trackbar_hw1(int, void*) {}
int nLineThinkness = 0;
int nRadius = 0;


void onChange(int nValue, void* pUserdata)
{
	int nAdd_Value = nValue - 130;
	cout << "추가 화소값" << nAdd_Value << endl;

	Mat temp = img_p30 + nAdd_Value;
	imshow(title, temp);
}

void drawCircle(int event, int x, int y, int flags, void* param)
{
	//3장 26page
#if 0
	if (event == EVENT_LBUTTONDOWN)
	{
		nDrawing_p26 = true;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (nDrawing_p26 == true)
		{
			circle(img_p26, Point(x, y), 3, Scalar(0, 0, 255), 10);
		}
	}
	else if (event == EVENT_LBUTTONUP)
	{
		nDrawing_p26 = false;
	}
#endif

	//3장 33page
#if 1
	if (event == EVENT_LBUTTONDOWN)
	{
		nDrawing_p33 = true;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		if (nDrawing_p33 == true)
		{
			circle(img_p33, Point(x, y), 3, Scalar(blue, green, red), 10);
		}
	}
	else if (event == EVENT_LBUTTONUP)
	{
		nDrawing_p33 = false;
	}
	imshow("img", img_p33);

#endif
}

void hwDraw(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat img = *(Mat*)(param);
		if (nLineThinkness < 1)
		{
			if (nRadius < 10)
			{
				circle(img, Point(x, y), 10, Scalar(0, 255, 0), 1);
			}
			else
			{
				circle(img, Point(x, y), nRadius, Scalar(0, 255, 0), 1);
			}
		}
		else
		{
			if (nRadius < 10)
			{
				circle(img, Point(x, y), 10, Scalar(0, 255, 0), nLineThinkness);
			}
			else
			{
				circle(img, Point(x, y), nRadius, Scalar(0, 255, 0), nLineThinkness);
			}
		}
		imshow("hw1", img);// 영상이 변경되면 다시 표시한다
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		Mat img = *(Mat*)(param);
		Point_<int> pt1(x, y);
		Rect rect(pt1, Size(100, 100));
		if (nLineThinkness <= 0)
		{
			rectangle(img, rect, blue, 2); //사각형 그리기
		}
		else
		{
			rectangle(img, rect, blue, nLineThinkness); //사각형 그리기
		}

		imshow("hw1", img);// 영상이 변경되면 다시 표시한다
	}
	else if (event == EVENT_LBUTTONUP)
	{
	}
	else if (event == EVENT_RBUTTONUP)
	{
	}
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	//3장 19page
#if 0
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		cout << "마우스 왼쪽버튼 누르기" << endl;
		break;
	case EVENT_RBUTTONDOWN:
		cout << "마우스 오른쪽버튼 누르기" << endl;
		break;
	case EVENT_LBUTTONUP:
		cout << "마우스 왼쪽버튼 떼기" << endl;
		break;
	case EVENT_RBUTTONUP:
		cout << "마우스 오른쪽버튼 떼기" << endl;
		break;
	}
#endif

	//3장 23page
#if 1
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat img = *(Mat*)(param);
		circle(img, Point(x, y), 200, Scalar(0, 255, 0), 10);
		putText(img, "I found a dog!", Point(x, y + 200), FONT_HERSHEY_PLAIN, 2.0, 255, 2);
		imshow("src ", img);// 영상이 변경되면 다시 표시한다
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
	}
	else if (event == EVENT_LBUTTONUP)
	{
	}
	else if (event == EVENT_RBUTTONUP)
	{
	}
#endif
}

void print_matInfo(string name, Mat img)
{
	string mat_type;
	if (img.depth() == CV_8U)
	{
		mat_type = "CV_8U";
	}
	else if (img.depth() == CV_8S)
	{
		mat_type = "CV_8S";
	}
	else if (img.depth() == CV_16U)
	{
		mat_type = "CV_16U";
	}
	else if (img.depth() == CV_16S)
	{
		mat_type = "CV_16S";
	}
	else if (img.depth() == CV_32S)
	{
		mat_type = "CV_32S";
	}
	else if (img.depth() == CV_32F)
	{
		mat_type = "CV_32F";
	}
	else if (img.depth() == CV_64F)
	{
		mat_type = "CV_64F";
	}
	cout << name;
	cout << format(": depth(%d) channels(%d) -> 자료형 : ", img.depth(), img.channels());
	cout << mat_type << "C" << img.channels() << endl;
}

//문자열 출력함수 - 그림자 효과
void put_string(Mat& frame, string text, Point pt, int value)
{
	text += to_string(value);
	Point shade = pt + Point(2, 2);
	int font = FONT_HERSHEY_SIMPLEX;
	putText(frame, text, shade, font, 0.7, Scalar(0, 0, 0), 2); //그림자효과
	putText(frame, text, pt, font, 0.7, Scalar(120, 200, 90), 2); // 작성문자
}

VideoCapture capture;
void zoom_bar(int value, void*)
{
	capture.set(CAP_PROP_ZOOM, value);
}
void focus_bar(int value, void*)
{
	capture.set(CAP_PROP_FOCUS, value);
}

Mat img, roi;
int mx1, my1, mx2, my2;
bool cropping = false;

void onMouse_4_33(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		mx1 = x;
		my1 = y;
		cropping = true;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		mx2 = x;
		my2 = y;
		cropping = false;
		rectangle(img, Rect(mx1, my1, mx2 - mx1, my2 - my1), Scalar(0, 255, 0), 2);
		imshow("image", img);
	}
}

void brighten(Mat& img, int value)
{
	for (int r = 0; r < img.rows; r++)
	{
		for (int c = 0; c < img.cols; c++)
		{
			img.at<uchar>(r, c) = saturate_cast<uchar>(img.at<uchar>(r, c) + value);
		}
	}
}

void Threshold_Demo(int, void*)
{
	threshold(src_gray, dst, threshold_value, 255, threshold_type);
	imshow("결과영상", dst);
}

int stretch(int x, int r1, int s1, int r2, int s2)
{
	float result;
	if (0 <= x && x <= r1)
	{
		result = s1 / r1 * x;
	}
	else if (r1 < x && x <= r2)
	{
		result = ((s2 - s1) / (r2 - r1)) * (x - r1) + s1;
	}
	else if (r2 < x && x <= 255)
	{
		result = ((255 - s2) / (255 - r2)) * (x - r2) + s2;
	}
	return (int)result;
}

void drawHist(int histogram[])
{
	int hist_w = 512; //히스토그램 영상의 폭
	int hist_h = 400; //히스토그램 영사의 높이
	int bin_w = cvRound((double)hist_w / 256); //빈의 폭

	//히스토그램이 그려지는 영상(칼라로 정의)
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	//히스토그램의 최대값을 찾는다.
	int max = histogram[0];
	for (int i = 1; i < 256; i++)
	{
		if (max < histogram[i])
		{
			max = histogram[i];
		}
	}

	//히스토그램 배열을 최대값으로 정규화 한다.(최대값이 최대높이가 되도록)
	for (int i = 0; i < 255; i++)
	{
		histogram[i] = floor(((double)histogram[i] / max) * histImage.rows);
	}

	//히스토그램의 값을 빨강색 막대로 그린다.
	for (int i = 0; i < 255; i++)
	{
		line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - histogram[i]), Scalar(0, 0, 255));
	}

	imshow("Histogram", histImage);
}

void calc_Histo(const Mat& image, Mat& hist, int bins, int range_max = 256)
{
	int histSize[] = { bins };
	float range[] = { 0, (float)range_max };
	int channels[] = { 0 };
	const float* ranges[] = { range };

	calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges);
}

void draw_Histo(Mat hist, Mat &hist_img, Size size = Size(256, 200))
{
	hist_img = Mat(size, CV_8U, Scalar(255));
	float bin = (float)hist_img.cols / hist.rows;
	normalize(hist, hist, 0, hist_img.rows, NORM_MINMAX);

	for (int i = 0; i < hist.rows; i++)
	{
		float start_x = i * bin;
		float end_x = (i + 1) * bin;
		Point2f pt1(start_x, 0);
		Point2f pt2(end_x, hist.at<float>(i));

		if(pt2.y > 0)
		{
			rectangle(hist_img, pt1, pt2, Scalar(0), -1);
		}
	}
	flip(hist_img, hist_img, 0);
}

void create_hist(Mat img, Mat& hist, Mat& hist_img)
{
	int histsize = 256, range = 256;
	calc_Histo(img, hist, histsize, range);
	draw_Histo(hist, hist_img);
}

void filter(Mat img, Mat& dst, Mat mask)
{
	dst = Mat(img.size(), CV_32F, Scalar(0));
	Point h_m = mask.size() / 2;

	for (int i = h_m.y; i < img.rows - h_m.y; i++)
	{
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			float sum = 0;
			for (int u = 0; u < mask.rows; u++) //마스크 원소 회순
			{
				for (int v = 0; v < mask.cols; v++)
				{
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					sum += mask.at<float>(u, v) * img.at<uchar>(y, x);//회선수식
				}
			}
			dst.at<float>(i, j) = sum;
		}
	}
}

void differential(Mat image, Mat& dst, float data1[], float data2[])
{
	Mat dst1, mask1(3, 3, CV_32F, data1);
	Mat dst2, mask2(3, 3, CV_32F, data2);

	filter2D(image, dst1, CV_32F, mask1);
	filter2D(image, dst2, CV_32F, mask2);
	magnitude(dst1, dst2, dst);
	dst.convertTo(dst, CV_8U);
	convertScaleAbs(dst1, dst1); //절대값 및 형 번환 동시 수행
	convertScaleAbs(dst2, dst2);
	imshow("dst1 - 수직 마스크", dst1);
	imshow("dst2 - 수평 마스크", dst2);
}


//공간필터링 p.44
Mat detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;

static void CannyThreshold(int, void*)
{
	blur(src, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ::ratio, kernel_size);
	dst = Scalar::all(0);
	src.copyTo(dst, detected_edges);
	imshow("Image", src);
	imshow("Canny", dst);
}

//기하학적변환 p.10
float Lerp(float s, float e, float t)
{
	return s + (e - s) * t;
}
float Blerp(float c00, float c10, float c01, float c11, float tx, float ty)
{
	return Lerp(Lerp(c00, c10, tx), Lerp(c01, c11, tx), ty);
}

float GetPixel(Mat img, int x, int y)
{
	if (x > 0 && y > 0 && x < img.cols && y < img.rows)
	{
		return (float)(img.at<uchar>(y, x));
	}
	else
	{
		return 0;
	}
}

//마스크 원소와 마스크 범위 입력화소 간의 일치 여부 체크
bool check_match(Mat img, Point start, Mat mask, int mode = 0)
{
	for (int u = 0; u < mask.rows; u++) 
	{
		for (int v = 0; v < mask.cols; v++)
		{
			Point pt(v, u);						//순회좌표
			int m = mask.at<uchar>(pt);			//마스크 계수
			int p = img.at<uchar>(start + pt);	//해당 위치 입력화소 

			bool ch = (p == 255);				//일치 여부 비교
			if (m == 1 && ch == mode)			//mode 0이면 침식, 1이면 팽창
			{
				return  false;
			}
		}
	}
	return true;
}

//침식 연산
void erosion(Mat img, Mat& dst, Mat mask)
{
	dst = Mat(img.size(), CV_8U, Scalar(0));
	if (mask.empty())	mask = Mat(3, 3, CV_8UC1, Scalar(0));

	Point h_m = mask.size() / 2; //마스크 절반 크기

	for (int i = h_m.y; i < img.rows - h_m.y; i++) 
	{
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			Point start = Point(j, i) - h_m;
			bool  check = check_match(img, start, mask, 0); //원소 일치여부 비교
			dst.at<uchar>(i, j) = (check) ? 255 : 0;		// 출력화소 저장
		}
	}
}

//팽창연산
void dilation(Mat img, Mat& dst, Mat mask)
{
	dst = Mat(img.size(), CV_8U, Scalar(0));
	if (mask.empty())	mask = Mat(3, 3, CV_8UC1, Scalar(0));

	Point h_m = mask.size() / 2;
	for (int i = h_m.y; i < img.rows - h_m.y; i++) 
	{
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			Point start = Point(j, i) - h_m;
			bool  check = check_match(img, start, mask, 1);	// 원소 일치여부 비교
			dst.at<uchar>(i, j) = (check) ? 0 : 255;			// 침식연산과 반대
		}
	}
}

//열림연산
void opening(Mat img, Mat& dst, Mat mask)
{
	Mat tmp;
	erosion(img, tmp, mask);
	dilation(tmp, dst, mask);
}

//닫힘연산
void closing(Mat img, Mat& dst, Mat mask)
{
	Mat tmp;
	dilation(img, tmp, mask);
	erosion(tmp, dst, mask);
}

void morphology(Mat img, Mat& dst, Mat mask, int mode)
{
	dst = Mat(img.size(), CV_8U, Scalar(0));
	if (mask.empty())	mask = Mat(3, 3, CV_8UC1, Scalar(0));

	Point h_m = mask.size() / 2;

	for (int i = h_m.y; i < img.rows - h_m.y; i++) 
	{
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			Point start = Point(j, i) - h_m;
			bool  check = check_match(img, start, mask, mode);

			if (mode == 0) dst.at<uchar>(i, j) = (check) ? 0 : 255;
			else if (mode == 1)	dst.at<uchar>(i, j) = (check) ? 255 : 0;
		}
	}
}

//컬러영상처리
void bgr2hsi(Mat img, Mat& hsv)
{
	hsv = Mat(img.size(), CV_32FC3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			float B = img.at<Vec3b>(i, j)[0];
			float G = img.at<Vec3b>(i, j)[1];
			float R = img.at<Vec3b>(i, j)[2];

			float s = 1 - 3 * min(R, min(G, B)) / (R + B + G);
			float v = (R + B + G) / 3.0f;

			float tmp1 = ((R - G) + (R - B)) * 0.5f;
			float tmp2 = sqrt((R - G) * (R - B) + (G - B) * (G - B));
			float angle = acos(tmp1 / tmp2) * (180.f / CV_PI);
			float h = (B <= G) ? angle : 360 - angle;
			hsv.at<Vec3f>(i, j) = Vec3f(h / 2, s * 255, v);
		}
	}
	hsv.convertTo(hsv, CV_8U);
}

//14 - 주파수영역처리 - p.10
void displayDFT(Mat& src)
{
	Mat image_array[2] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	// ① DFT 결과 영상을 2개의 영상으로 분리한다. 
	split(src, image_array);

	Mat mag_image;
	// ② 푸리에 변환 계수들의 절대값을 계산한다. 
	magnitude(image_array[0], image_array[1], mag_image);

	// ③ 푸리에 변환 계수들은 상당히 크기 때문에 로그 스케일로 변환한다. 
	// 0값이 나오지 않도록 1을 더해준다. 
	mag_image += Scalar::all(1);
	log(mag_image, mag_image);

	// ④ 0에서 255로 범위로 정규화한다. 
	normalize(mag_image, mag_image, 0, 1, NORM_MINMAX);
	imshow("DFT", mag_image);
}

void shuffleDFT(Mat& src)
{
	int cX = src.cols / 2;
	int cY = src.rows / 2;
	
	Mat q1(src, Rect(0, 0, cX, cY));
	Mat q2(src, Rect(cX, 0, cX, cY));
	Mat q3(src, Rect(0, cY, cX, cY));
	Mat q4(src, Rect(cX, cY, cX, cY));

	Mat tmp;
	q1.copyTo(tmp);
	q4.copyTo(q1);
	tmp.copyTo(q4);
	q2.copyTo(tmp);
	q3.copyTo(q2);
	tmp.copyTo(q3);
}

//14 - 주파수영역처리 - p.17
// 원형 필터를 만든다.(저주파)
Mat getFilter_Circle(Size size)
{
	Mat filter(size, CV_32FC2, Vec2f(0, 0));
	circle(filter, size / 2, 50, Vec2f(1, 1), -1);
	return filter;
}

//14 - 주파수영역처리 - p.21
// 버터워쓰 필터를 만든다. 
Mat getFilter(Size size)
{
	Mat tmp = Mat(size, CV_32F);
	Point center = Point(tmp.rows / 2, tmp.cols / 2);
	double radius;
	double D = 50;
	double n = 2;

	for (int i = 0; i < tmp.rows; i++) {
		for (int j = 0; j < tmp.cols; j++) {
			radius = (double)sqrt(pow((i - center.x), 2.0) + pow((double)(j - center.y), 2.0));
			tmp.at<float>(i, j) = (float)
				(1 / (1 + pow((double)(radius / D), (double)(2 * n))));
		}
	}
	Mat toMerge[] = { tmp, tmp };
	Mat filter;
	merge(toMerge, 2, filter);
	return filter;
}

Mat getFilter_Pattern(Size size)
{
	Mat tmp = Mat(size, CV_32F);

	for (int i = 0; i < tmp.rows; i++) 
	{
		for (int j = 0; j < tmp.cols; j++) 
		{
			if (j > (tmp.cols / 2 - 10) && j<(tmp.cols / 2 + 10) && i >(tmp.rows / 2 + 10))
			{
				tmp.at<float>(i, j) = 0;
			}
			else if (j > (tmp.cols / 2 - 10) && j < (tmp.cols / 2 + 10) && i < (tmp.rows / 2 - 10))
			{
				tmp.at<float>(i, j) = 0;
			}
			else
			{
				tmp.at<float>(i, j) = 1;
			}
		}
	}
	Mat toMerge[] = { tmp, tmp };
	Mat filter;
	merge(toMerge, 2, filter);
	return filter;
}

int threshold_value_Image_Seg = 128;
int threshold_type_Image_Seg = 0;
const int max_value_Image_Seg = 255;
const int max_binary_value_Image_Seg = 255;
Mat src_Image_Seg, src_gray_Image_Seg, dst_Image_Seg;

static void MyThreshold(int, void*)
{
	threshold(src_Image_Seg, dst_Image_Seg, threshold_value_Image_Seg, max_binary_value_Image_Seg, threshold_type_Image_Seg);
	imshow("result", dst_Image_Seg);
}

//전처리
Mat  preprocessing(Mat img)
{
	Mat gray, th_img;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7, 7), 2, 2);

	threshold(gray, th_img, 130, 255, THRESH_BINARY | THRESH_OTSU);
	morphologyEx(th_img, th_img, MORPH_OPEN, Mat(), Point(-1, -1), 1);

	return th_img;
}

// 검출 영역 원좌표로 반환 
vector<RotatedRect> find_coins(Mat img)
{
	vector<vector<Point> > contours;
	findContours(img.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<RotatedRect> circles;
	for (int i = 0; i < (int)contours.size(); i++)
	{
		RotatedRect  mr = minAreaRect(contours[i]);
		mr.angle = (mr.size.width + mr.size.height) / 4.0f;

		if (mr.angle > 18)
		{
			circles.push_back(mr);
		}
	}
	return circles;
}

void setLabel(Mat& img, const vector<Point>& pts, const String& label)
{
	Rect rc = boundingRect(pts);
	rectangle(img, rc, Scalar(0, 0, 255), 1);
	putText(img, label, rc.tl(), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
}

void cornerharris(Mat image, Mat& corner, int bSize, int ksize, float k)
{
	Mat dx, dy, dxy, dx2, dy2;
	corner = Mat(image.size(), CV_32F, Scalar(0));

	Sobel(image, dx, CV_32F, 1, 0, ksize);
	Sobel(image, dy, CV_32F, 0, 1, ksize);
	multiply(dx, dx, dx2);
	multiply(dy, dy, dy2);
	multiply(dx, dy, dxy);

	Size msize(5, 5);
	GaussianBlur(dx2, dx2, msize, 0);
	GaussianBlur(dy2, dy2, msize, 0);
	GaussianBlur(dxy, dxy, msize, 0);

	for (int i = 0; i < image.rows; i++) 
	{
		for (int j = 0; j < image.cols; j++)
		{
			float  a = dx2.at<float>(i, j);
			float  b = dy2.at<float>(i, j);
			float  c = dxy.at<float>(i, j);
			corner.at<float>(i, j) = (a * b - c * c) - k * (a + b) * (a + b);
		}
	}
}

Mat draw_coner(Mat corner, Mat image, int thresh)
{
	int cnt = 0;
	normalize(corner, corner, 0, 100, NORM_MINMAX, CV_32FC1, Mat());

	for (int i = 1; i < corner.rows - 1; i++) 
	{
		for (int j = 1; j < corner.cols - 1; j++)
		{
			float cur = (int)corner.at<float>(i, j);
			if (cur > thresh)
			{
				if (cur > corner.at<float>(i - 1, j)
					&& cur > corner.at<float>(i + 1, j)
					&& cur > corner.at<float>(i, j - 1)
					&& cur > corner.at<float>(i, j + 1))
				{
					circle(image, Point(j, i), 2, Scalar(255, 0, 0), -1);
					cnt++;
				}
			}
		}
	}
	cout << "코너 개수: " << cnt << endl;
	return image;
}

Mat image, corner1, corner2;

void cornerHarris_demo(int  thresh, void*)
{
	Mat img1 = draw_coner(corner1, image.clone(), thresh);
	Mat img2 = draw_coner(corner2, image.clone(), thresh);

	imshow("img1-User harris", img1);
	imshow("img2-OpenCV harris", img2);
}

void corner_fast()
{
	Mat src = imread("D:\\999.Image\\building.jpg", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cout << "Image Load failed!" << endl;
		return;
	}

	vector<KeyPoint> keypoints;
	FAST(src, keypoints, 60, true);

	Mat dst;
	cvtColor(src, dst, COLOR_GRAY2BGR);
	for (KeyPoint kp : keypoints)
	{
		Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
		circle(dst, pt, 5, Scalar(0, 0, 255), 2);
	}
	imshow("src", src);
	imshow("dst", dst);

	waitKey();
}

void detect_keypoints()
{
	Mat src = imread("D:\\999.Image\\box_in_scene.png", IMREAD_GRAYSCALE);

	if (src.empty())
	{
		cout << "Image Load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints;
	feature->detect(src, keypoints);

	Mat desc;
	feature->compute(src, keypoints, desc);

	cout << "ketpoints.size() : " << keypoints.size() << endl;
	cout << "desc.size() : " << desc.size() << endl;

	Mat dst;
	drawKeypoints(src, keypoints, dst, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("src", src);
	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void template_matching()
{
	Mat img = imread("D:\\999.Image\\circuit.bmp", IMREAD_COLOR);

	Mat templ = imread("D:\\999.Image\\crystal.bmp", IMREAD_COLOR);

	if (img.empty() || templ.empty())
	{
		cerr << " Image load failed!" << endl;
		return;
	}

	img = img + Scalar(50, 50, 50);

	Mat noise(img.size(), CV_32SC3);
	randn(noise, 0, 10);
	add(img, noise, img, Mat(), CV_8UC3);

	Mat res, res_norm;
	matchTemplate(img, templ, res, TM_CCOEFF_NORMED);
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);

	double maxv;
	Point maxloc;
	minMaxLoc(res, 0, &maxv, 0, &maxloc);
	cout << "maxv : " << maxv << endl;

	rectangle(img, Rect(maxloc.x, maxloc.y, templ.cols, templ.rows), Scalar(0, 0, 255), 2);

	imshow("templ", templ);
	imshow("res_norm", res_norm);
	imshow("img", img);
	waitKey();
	destroyAllWindows();
}

void keypoint_matching()
{
	Mat src1 = imread("D:\\999.Image\\box.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("D:\\999.Image\\box_in_scene.png", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty())
	{
		cerr << " Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> feature = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	feature->detectAndCompute(src1, Mat(), keypoints1, desc1);
	feature->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, matches, dst);

	imshow("dst", dst);

	waitKey();
	destroyAllWindows();
}

void find_homography()
{
	Mat src1 = imread("D:\\999.Image\\box.png", IMREAD_GRAYSCALE);
	Mat src2 = imread("D:\\999.Image\\box_in_scene.png", IMREAD_GRAYSCALE);

	if (src1.empty() || src2.empty())
	{
		cerr << " Image load failed!" << endl;
		return;
	}

	Ptr<Feature2D> orb = ORB::create();

	vector<KeyPoint> keypoints1, keypoints2;
	Mat desc1, desc2;
	orb->detectAndCompute(src1, Mat(), keypoints1, desc1);
	orb->detectAndCompute(src2, Mat(), keypoints2, desc2);

	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	std::sort(matches.begin(), matches.end());

	vector<DMatch> good_matches(matches.begin(), matches.begin() + 50);

	Mat dst;
	drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
		Scalar::all(-1), Scalar::all(-1), vector<char>(), 
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	vector<Point2f> pts1, pts2;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		pts1.push_back(keypoints1[good_matches[i].queryIdx].pt);
		pts2.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(pts1, pts2, RANSAC);

	vector<Point2f> corner1, corner2;
	corner1.push_back(Point2f(0, 0));
	corner1.push_back(Point2f(src1.cols - 1.f, 0));
	corner1.push_back(Point2f(src1.cols - 1.f, src1.rows - 1.f));
	corner1.push_back(Point2f(0, src1.rows - 1.f));
	perspectiveTransform(corner1, corner2, H);

	vector<Point> corners_dst;
	for (Point2f pt : corner2)
	{
		corners_dst.push_back(Point(cvRound(pt.x + src1.cols), cvRound(pt.y)));
	}

	polylines(dst, corners_dst, true, Scalar(0, 255, 0), 2, LINE_AA);

	imshow("dst", dst);
	waitKey();
	destroyAllWindows();
}


int main()
{
	//1주
#if 0
	 Mat Image;
	Image = imread("D:\\999.Image\\lenna.jpg", IMREAD_COLOR);

	if (Image.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}

	imshow("src", Image);

	Mat gray, edge, output;
	cvtColor(Image, gray, COLOR_BGR2GRAY);

	imshow("gray", gray);

	imwrite("D:\\gray.jpg", gray);

	waitKey(0);
#endif
	//2장
	//8 Page
#if 0
	//Point_ 객체 선언 방식
	Point_<int> pt1(100, 200);
	Point_<float> pt2(92.3f, 125.23f);
	Point_<double> pt3(100.2, 300.9);

	//Point_ 객체 간결 선언 방식
	Point2i pt4(120, 69);
	Point2f pt5(0.3f, 0.f), pt6(0.f, 0.4f);
	Point2d pt7(0.25, 0.6);

	//Pint_ 객체연산
	Point pt8 = pt1 + (Point)pt2; //자료형이 다른 Point 객체 덧셈
	Point2f pt9 = pt6 * 3.14f;
	Point2d pt10 = (pt3 + (Point2d)pt6) * 10;
	cout << "pt8 = " << pt8.x << " , " << pt8.y << endl;
	cout << "[pt9] = " << pt9 << endl;
	cout << "(pt2 == pt6)" << endl;
	cout << "pt7과 pt8의 내적 :" << pt7.dot(pt8) << endl;
#endif

	//13 Page
#if 0
	//Size_ 객체 선언 방식
	Size_<int> sz1(100, 200);
	Size_<float> sz2(192.3f, 25.3f);
	Size_<double> sz3(100.2, 30.9);

	//Size_ 객체 간결 선언 방식6
	Size sz4(120, 69);
	Size2f sz5(0.3f, 0.f);
	Size2d sz6(0.25, 0.6);

	Point2d pt1(0.25, 0.6);

	Size2i sz7 = sz1 + (Size2i)sz2;
	Size2d sz8 = sz3 - (Size2d)sz4;
	Size2d sz9 = sz5 + (Size2f)pt1;

	cout << "sz1.width = " << sz1.width;
	cout << ", sz1.height = " << sz1.height << endl;
	cout << "sz1 넓이 :  " << sz1.area() << endl;
	cout << "[sz7] = " << sz7 << endl;
	cout << "[sz8] = " << sz8 << endl;
	cout << "[sz9] = " << sz9 << endl;

#endif

	//19 Page
#if 0
	Size2d sz(100.5, 60.6);
	Point2f pt1(20.f, 30.f), pt2(100.f, 200.f);

	//Rect_ 객체 기본 선언 방식 
	Rect_<int> rect1(10, 10, 30, 50);
	Rect_<float> rect2(pt1, pt2);
	Rect_<double> rect3(Point2d(20.5, 10), sz);

	//Rect 간결 선업 방식 & 연산 적용
	Rect rect4 = rect1 + (Point)pt1;
	Rect2f rect5 = rect2 + (Size2f)sz;
	Rect2d rect6 = rect1 & (Rect)rect2;

	//결과 
	cout << "rect3 = " << rect3.x << ", " << rect3.y << ", ";
	cout << rect3.width << " x " << rect3.height << endl;
	cout << "rect4 = " << rect4.tl() << " " << rect4.br() << endl;
	cout << "rect5 크기 = " << rect5.size() << endl;
	cout << "[rect6]  = " << rect6 << endl;
#endif

	//21 Page
#if 0
	//기본선언 및 간결 방법
	Vec <int, 2> v1(5, 12);
	Vec <double, 3> v2(40, 130.7, 125.6);
	Vec2d v3(10, 10);
	Vec6f v4(40.f, 230.25f, 525.6f);
	Vec3i v5(200, 230, 250);

	//객체 연산 및 형변환
	Vec3d v6 = v2 + (Vec3d)v5;
	Vec2d v7 = (Vec2d)v1 + v3;
	Vec6f v8 = v4 * 20.0f;

	Point pt1 = v1 + (Vec2i)v7;
	Point3_<int> pt2 = static_cast<Point3_<int>>(v2);

	//콘솔창 출력
	cout << "[v3] = " << v3 << endl;
	cout << "[v7] = " << v7 << endl;
	cout << "[v3 * v7] = " << v3.mul(v7) << endl;
	cout << "v8[0] = " << v8[0] << endl;
	cout << "v8[1] = " << v8[1] << endl;
	cout << "v8[2] = " << v8[2] << endl;
	cout << "[v2] = " << v2 << endl;
	cout << "[pt2] = " << pt2 << endl;
#endif

	//24 Page
#if 0
	Scalar_<uchar> red(0, 0, 255);
	Scalar_<int> blue(255, 0, 0);
	Scalar_<double> color1(500);
	Scalar_<float> color2(100.f, 200.f, 125.9f);

	Vec3d green(0, 0, 300.5);
	Scalar green1 = color1 + (Scalar)green;
	Scalar green2 = color2 + (Scalar_<float>)green;

	cout << "blue = " << blue[0] << ", " << blue[1];
	cout << ", " << blue[1] << ", " << blue[2] << endl;
	cout << "red = " << red << endl;
	cout << "green = " << green << endl << endl;
	cout << "green1 = " << green1 << endl;
	cout << "green2 = " << green2 << endl;
#endif

	//27 Page
#if 0
	Scalar blue(255, 0, 0), red(0, 0, 255), green(0, 255, 0); //색상선언
	Scalar white = Scalar(255, 255, 255); //흰색 색상
	Scalar yellow(0, 255, 255);

	Mat image(400, 600, CV_8UC3, white);
	Point pt1(50, 130), pt2(200, 300), pt3(300, 150), pt4(400, 50); //좌표선언
	Rect rect(pt3, Size(200, 150));

	line(image, pt1, pt2, red);//직선그리기
	line(image, pt3, pt4, green, 2, LINE_AA);//안티에일리싱 선
	line(image, pt3, pt4, green, 3, LINE_8, 1);//8방향 연결선, 1비트 시프트

	rectangle(image, rect, blue, 2); //사각형 그리기
	rectangle(image, rect, blue, FILLED, LINE_4, 1); //4방향 연결선, 1비트 시프트
	rectangle(image, pt1, pt2, red, 3);

	imshow("직선 & 사각형", image);
	waitKey(0);

#endif

	//34 Page
#if 0
	Scalar orange(0, 165, 255), blue(255, 0, 0), magenta(255, 0, 255); //색상선언
	Mat image(300, 500, CV_8UC3, Scalar(255, 255, 255));

	Point center = (Point)image.size() / 2; //영상중심좌표
	Point pt1(70, 50), pt2(350, 220);

	circle(image, center, 100, blue);
	circle(image, pt1, 80, orange, 2);
	circle(image, pt2, 60, magenta, -1);

	int font = FONT_HERSHEY_COMPLEX;
	putText(image, "center_blue", center, font, 1.2, blue);
	putText(image, "pt1_orange", pt1, font, 0.8, orange);
	putText(image, "pt2_magenta", pt2 + Point(2, 2), font, 0.5, Scalar(0, 0, 0), 2);
	putText(image, "pt2_magenta", pt2, font, 0.5, magenta, 1);

	imshow("원그리기", image);
	waitKey(0);
#endif

	//39 Page
#if 0
	//
	//검정색으로 초기화된 600 × 400 크기의 영상 생성
	Mat image = Mat(400, 600, CV_8UC3, Scalar(0, 0, 0));
	line(image, Point(100, 100), Point(300, 300), Scalar(0, 0, 255), 7);
	rectangle(image, Point(250, 30), Point(450, 200), Scalar(0, 255, 0), 5);
	circle(image, Point(100, 300), 60, Scalar(255, 0, 0), 3);
	ellipse(image, Point(300, 350), Point(100, 60), 45, 130, 270, Scalar(255, 255, 255), 5);
	imshow("Image", image);
	waitKey(0);
#endif

	//HW1
#if 0
	Scalar blue(255, 0, 0); //색상선언
	Mat image = Mat(800, 600, CV_8UC3, Scalar(0, 0, 0));

	circle(image, Point(100, 100), 50, blue);
	imshow("Image", image);
	waitKey(0);

#endif

	//HW2
#if 0
	Mat image = Mat(400, 600, CV_8UC3, Scalar(255, 255, 255));
	int nDiameter = 200;
	Scalar blue(255, 0, 0), red(0, 0, 255); // 색상
	Point center(image.cols / 2, image.rows / 2); // 태극 중심 좌표
	Size circle_Big(nDiameter / 2, nDiameter / 2), circle_small(nDiameter / 4, nDiameter / 4); // 태극 문양을 그리기 위한 두개의 원

	// 빨강과 파란색의 반원을 아래위로 그린다.
	ellipse(image, center, circle_Big, 0, 0, 180, blue, -1);
	ellipse(image, center, circle_Big, 0, 180, 360, red, -1);
	// 작은 반원을 아래 위로 그려준다.
	ellipse(image, Point(center.x - circle_small.width, center.y), circle_small, 0, 0, 180, red, -1);
	ellipse(image, Point(center.x + circle_small.width, center.y), circle_small, 0, 180, 360, blue, -1);

	imshow("Image", image);
	waitKey(0);
#endif

	//3장
	//4 page
#if 0
	Mat image1(300, 400, CV_8U, Scalar(255));
	Mat image2(300, 400, CV_8U, Scalar(100));

	string title1 = "white창 제어";
	string title2 = "gray창 제어";

	namedWindow(title1, WINDOW_AUTOSIZE);
	namedWindow(title2, WINDOW_NORMAL);
	moveWindow(title1, 100, 200);
	moveWindow(title2, 300, 200);

	imshow(title1, image1);
	imshow(title2, image2);
	waitKey();
	destroyAllWindows();
#endif

	//9 page
#if 0
	Mat image(200, 300, CV_8U, Scalar(255));
	namedWindow("키보드 이벤트", WINDOW_AUTOSIZE);
	imshow("키보드 이벤트", image);

	while (1)
	{
		int nkey = waitKey(100);
		if (nkey == 27)
		{
			break;
		}

		switch (nkey)
		{
		case 'a':
			cout << "a키 입력" << endl;
			break;
		case 'b':
			cout << "b키 입력" << endl;
			break;
		case 0x41:
			cout << "A키 입력" << endl;
			break;
		case 66:
			cout << "B키 입력" << endl;
			break;
		case 0x250000:
			cout << "왼쪽 화살표 키 입력" << endl;
			break;
		case 0x260000:
			cout << "윗쪽 화살표 키 입력" << endl;
			break;
		case 0x270000:
			cout << "오른쪽 화살표 키 입력" << endl;
			break;
		case 0x280000:
			cout << "아래쪽 화살표 키 입력" << endl;
			break;
		}

	}
#endif

	//12 page
#if 0
	Mat img;
	img = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\2.2주차실습\\3.Image\\dog.jpg", IMREAD_COLOR);
	if (img.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}

	imshow("img", img);
	int x = 300;
	int y = 300;

	while (1)
	{
		int key = waitKey(100);
		if (key == 'q')
		{
			break;
		}
		else if (key == 'a')
		{
			x -= 10;
		}

		else if (key == 'w')
		{
			y -= 10;
		}
		else if (key == 'd')
		{
			x += 10;
		}
		else if (key == 's')
		{
			y += 10;
		}
		circle(img, Point(x, y), 200, Scalar(0, 255, 0), 5);
		imshow("img", img);
	}
#endif

	//15 page
#if 0
	Mat	src = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\2.2주차실습\\3.Image\\photo1.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}
	imshow("src", src);
	while (1)
	{
		int	key = waitKeyEx();// 사용자로부터 키를 기다린다.
		cout << key << " ";

		if (key == 'q')// 사용자가 ‘q' 를 누르면 종료한다.
		{
			break;
		}
		else if (key == 2424832)
		{
			// 왼쪽화살표 키
			src -= 50;// 영상이 어두워진다
		}
		else if (key == 2555904)
		{
			// 오른쪽화살표 키
			src += 50;// 영상이 밝아진다.
		}
		imshow("src ", src);// 영상이 변경되었으므로 다시 표시한다
	}
#endif

	//19 page		
#if 0
	Mat image(200, 300, CV_8U);
	image.setTo(255);
	imshow("Mouse Event1", image);
	imshow("Mouse Event2", image);

	setMouseCallback("Mouse Event1", onMouse, 0);
	waitKey(0);
#endif

	//23 page		
#if 0
	Mat	src = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\2.2주차실습\\3.Image\\dog.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}
	imshow("src", src);

	setMouseCallback("src", onMouse, &src);
	waitKey(0);
#endif

	//26 page		
#if 0
	img_p26 = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\2.2주차실습\\3.Image\\bug.jpg", IMREAD_COLOR);
	if (img_p26.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}
	imshow("Image", img_p26);
	setMouseCallback("Image", drawCircle);
	waitKey(0);

	imwrite("d:\\test.jpg", img_p26);
#endif

	//30 page		
#if 0
	int nValue = 128;
	img_p30 = Mat(300, 400, CV_8UC1, Scalar(120));

	namedWindow(title, WINDOW_AUTOSIZE);
	createTrackbar("밝기값", title, &nValue, 255, onChange);

	imshow(title, img_p30);
	waitKey(0);
#endif

	//33 page		
#if 0
	img_p33 = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\2.2주차실습\\3.Image\\bug.jpg", IMREAD_COLOR);
	if (img_p33.empty())
	{
		cout << "영상을 읽을 수 없음" << endl;
	}
	namedWindow("img", WINDOW_AUTOSIZE);
	imshow("img", img_p33);
	setMouseCallback("img", drawCircle);
	createTrackbar("R", "img", &red, 255, on_trackbar);
	createTrackbar("G", "img", &green, 255, on_trackbar);
	createTrackbar("B", "img", &blue, 255, on_trackbar);

	waitKey(0);
#endif

	//3장 HW1
#if 0
	img_hw1 = Mat(800, 600, CV_8UC1, Scalar(120));
	namedWindow("hw1", WINDOW_AUTOSIZE);
	imshow("hw1", img_hw1);
	setMouseCallback("hw1", hwDraw, &img_hw1);
	createTrackbar("Line", "hw1", &nLineThinkness, 10, on_trackbar_hw1);
	createTrackbar("Radius", "hw1", &nRadius, 200, on_trackbar_hw1);

	waitKey(0);
#endif
	//3 - p4~5
#if 0
	string filename = "D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\read_color.jpg";
	Mat color2gray = imread(filename, IMREAD_GRAYSCALE);
	Mat color2color = imread(filename, IMREAD_COLOR);
	CV_Assert(color2gray.data && color2color.data);

	Rect roi(100, 100, 1, 1);
	cout << "행렬 좌표 (100,100) 화소값 " << endl;
	cout << "color2gray " << color2gray(roi) << endl;
	cout << "color2color " << color2color(roi) << endl;

	print_matInfo("color2gray", color2gray);
	print_matInfo("color2color", color2color);
	imshow("color2gray", color2gray);
	imshow("color2color", color2color);

	waitKey(0);
#endif

	//3 - p8
#if 0
	Mat img8 = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\read_color.jpg", IMREAD_COLOR);
	CV_Assert(img8.data);

	vector<int> params_jpg, params_png;
	params_jpg.push_back(IMWRITE_JPEG_QUALITY);
	params_jpg.push_back(50);
	params_png.push_back(IMWRITE_PNG_COMPRESSION);
	params_png.push_back(9);

	imwrite("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\wrtie_test1.jpg", img8);
	imwrite("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\wrtie_test2.jpg", img8, params_jpg);
	imwrite("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\wrtie_test.png", img8, params_png);
	imwrite("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\wrtie_test.bmp", img8);
	waitKey(0);
#endif
	//3 - p14~15
#if 0
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "카메라가 연결 되지 않았습니다." << endl;
		exit(1);
	}

	//카메라 속성획득
	cout << "너비 : " << capture.get(CAP_PROP_FRAME_WIDTH) << endl;
	cout << "높이 : " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
	cout << "노출 : " << capture.get(CAP_PROP_EXPOSURE) << endl;
	cout << "밝기 : " << capture.get(CAP_PROP_BRIGHTNESS) << endl;

	for (;;)
	{
		Mat frame;
		capture.read(frame);

		put_string(frame, "EXPOS : ", Point(10, 4), capture.get(CAP_PROP_EXPOSURE));
		imshow("카메라 영상 보기", frame);
		if (waitKey(30) >= 0) break;
	}
#endif
	//3 - p17~18
#if 0
	capture.open(0);
	CV_Assert(capture.isOpened());

	capture.set(CAP_PROP_FRAME_WIDTH, 400);
	capture.set(CAP_PROP_FRAME_HEIGHT, 300);
	capture.set(CAP_PROP_AUTOFOCUS, 0);
	capture.set(CAP_PROP_BRIGHTNESS, 150);

	int zoom = capture.get(CAP_PROP_ZOOM);
	int focus = capture.get(CAP_PROP_FOCUS);

	string title = "카메라 속성 변경";
	namedWindow(title);
	createTrackbar("zoom", title, &zoom, 10, zoom_bar);
	createTrackbar("foxus", title, &focus, 40, focus_bar);

	for (;;)
	{
		Mat frame;
		capture.read(frame);

		put_string(frame, "zoom : ", Point(10, 240), zoom);
		put_string(frame, "foxus : ", Point(10, 270), focus);
		imshow(title, frame);
		if (waitKey(30) >= 0) break;
	}
#endif
	//3 - p20~21
#if 0
	VideoCapture capture(0);
	if (!capture.isOpened());

	double fps = 29.97;
	int delay = cvRound(1000.0 / fps);
	Size size(640, 360);
	int fourcc = VideoWriter::fourcc('D', 'X', '5', '0');

	capture.set(CAP_PROP_FRAME_WIDTH, size.width);
	capture.set(CAP_PROP_FRAME_HEIGHT, size.height);

	cout << "width x height : " << size << endl;
	cout << "VideoWriter::fourcc : " << fourcc << endl;
	cout << "dealy : " << delay << endl;
	cout << "fps : " << fps << endl;

	VideoWriter writer;//동영상파일 저장 객체

	//파일 개발 및 설정
	writer.open("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\video_file1.avi", fourcc, fps, size);
	CV_Assert(writer.isOpened());

	for (;;)
	{
		Mat frame;
		capture >> frame; //카메라영상받기
		writer << frame; //프레임을 도영ㅇ상으로 저장 

		imshow("카메라 영상받기", frame);
		if (waitKey(delay) >= 0)
		{
			break;
		}
	}
#endif
	//3 - p23
#if 0
	VideoCapture capture;
	capture.open("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\video_file.avi");
	CV_Assert(capture.isOpened());

	double frame_rate = capture.get(CAP_PROP_FPS);
	int delay = 1000 / frame_rate;
	int frame_cnt = 0;
	Mat frame;

	while (capture.read(frame))
	{
		if (waitKey(delay) >= 0) break;

		if (frame_cnt < 100);
		else if (frame_cnt < 200) frame -= Scalar(0, 0, 100);
		else if (frame_cnt < 300) frame += Scalar(0, 0, 100);
		else if (frame_cnt < 400) frame = frame * 1.5;
		else if (frame_cnt < 500) frame = frame * 0.5;

		put_string(frame, "frame_cnt", Point(20, 50), frame_cnt);
		imshow("동영상 파일 읽기", frame);
		frame_cnt++;
	}
#endif
	//3 - p26
#if 0
	VideoCapture cap("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\video_file.avi");
	if (!cap.isOpened())
	{
		cout << "동영상을 읽을 수 없음" << endl;
	}
	namedWindow("frame", 1);
	for (;;)
	{
		Mat frame;
		cap >> frame;
		imshow("frame", frame);
		if (waitKey(30) >= 0) break;
	}
#endif
	//4 - p6~7
#if 0
	float data[] = { 1.2f, 2.3f, 3.2f,
					 4.5f, 5.f, 6.5f };

	Mat m1(2, 3, CV_8U);
	Mat m2(2, 3, CV_8U, Scalar(300));
	Mat m3(2, 3, CV_16S, Scalar(300));
	Mat m4(2, 3, CV_32F, data);

	Size sz(2, 3);

	Mat m5(Size(2, 3), CV_64F);
	Mat m6(sz, CV_32F, data);

	cout << "[m1] = " << endl << m1 << endl;
	cout << "[m2] = " << endl << m2 << endl;
	cout << "[m3] = " << endl << m3 << endl;
	cout << "[m4] = " << endl << m4 << endl << endl;
	cout << "[m5] = " << endl << m5 << endl;
	cout << "[m6] = " << endl << m6 << endl;
#endif
	//4 - p17
#if 0
	Mat img = imread("D:\\1.개인폴더\\2.산업인공지능학과\\2.23년2학기(석사2학기)\\2.영상처리실제\\3.실습\\3.3주차실습\\3.Image\\lenna.jpg");
	if (img.empty()) { cout << "영상을 읽을 수 없음" << endl; return -1; }
	imshow("img", img);

	cout << "행의 수 = " << img.rows << endl;
	cout << "열의 수 = " << img.cols << endl;
	cout << "행렬의 크기 = " << img.size() << endl;
	cout << "전체 화소 개수 = " << img.total() << endl;
	cout << "한 화소 크기 = " << img.elemSize() << endl;
	cout << "타입 = " << img.type() << endl;
	cout << "채널 = " << img.channels() << endl;

	waitKey(0);
#endif
	//4 - p21
#if 0
	Mat m1(2, 3, CV_8U, 2);
	Mat m2(2, 3, CV_8U, Scalar(10));

	Mat m3 = m1 + m2;
	Mat m4 = m2 - 6;
	Mat m5 = m1;

	cout << "[m2] = " << endl << m2 << endl;
	cout << "[m3] = " << endl << m3 << endl;
	cout << "[m4] = " << endl << m4 << endl;

	cout << "[m1] = " << endl << m1 << endl;
	cout << "[m5] = " << endl << m5 << endl;
	m5 = 100;
	cout << "[m1] = " << endl << m1 << endl;
	cout << "[m5] = " << endl << m5 << endl;
#endif
	//4 - p33~34
#if 0
	img = imread("D:\\999.Image\\lenna.jpg");
	imshow("image", img);
	Mat clone = img.clone();
	setMouseCallback("image", onMouse_4_33);

	while (1)
	{
		int key = waitKey(100);
		if (key == 'q')
		{
			break;
		}
		else if (key == 'c')
		{
			roi = clone(Rect(mx1, my2, mx2 - mx1, my2 - my1));
			imwrite("d:\\result.jpg", roi);
		}
	}
#endif


	//5 - p8
#if 0
	Mat ch0(3, 4, CV_8U, Scalar(10));
	Mat ch1(3, 4, CV_8U, Scalar(20));
	Mat ch2(3, 4, CV_8U, Scalar(30));

	Mat bgr_arr[] = { ch0, ch1, ch2 };
	Mat bgr;
	merge(bgr_arr, 3, bgr);
	vector<Mat> bgr_vec;
	split(bgr, bgr_vec);

	cout << "[ch0] = " << endl << ch0 << endl;
	cout << "[ch1] = " << endl << ch1 << endl;
	cout << "[ch2] = " << endl << ch2 << endl << endl;

	cout << "[bgr] = " << endl << bgr << endl << endl;
	cout << "[bgr_vec[0] = " << endl << bgr_vec[0] << endl;
	cout << "[bgr_vec[1] = " << endl << bgr_vec[1] << endl;
	cout << "[bgr_vec[2] = " << endl << bgr_vec[2] << endl;

#endif
	//5 - p16
#if 0
	Mat image1(300, 300, CV_8U, Scalar(0));
	Mat image2(300, 300, CV_8U, Scalar(0));
	Mat image3, image4, image5, image6;

	Point center = image1.size() / 2;
	circle(image1, center, 100, Scalar(255), -1);
	rectangle(image2, Point(0, 0), Point(150, 300), Scalar(255), -1);

	bitwise_or(image1, image2, image3);
	bitwise_and(image1, image2, image4);
	bitwise_xor(image1, image2, image5);
	bitwise_not(image1, image6);

	imshow("image1", image1);
	imshow("image2", image2);
	imshow("bitwise_or", image3);
	imshow("bitwise_and", image4);
	imshow("bitwise_xor", image5);
	imshow("bitwise_not", image6);

	waitKey(0);
#endif
	//화소처리 - p.8
#if 0
	Mat	img = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	imshow("Original Image", img);

	brighten(img, 30);
	imshow("New Image", img);
	waitKey(0);
#endif
	//화소처리 - p.10
#if 0
	Mat	img = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	imshow("Original Image", img);

	for (int r = 0; r < img.rows; r++)
	{
		uchar* p = img.ptr<uchar>(r);
		
		for (int c = 0; c < img.cols; ++c)
		{
			p[c] = saturate_cast<uchar> (p[c] + 30);
		}
	}
	imshow("New Image", img);
	
	waitKey(0);
#endif
	//화소처리 - p.14
#if 0
	double alpha = 1.0;
	int beta = 0;
	Mat image = imread("D:\\999.Image\\contrast.jpg");

	Mat	oimage;
	cout << "알파값을 입력하시오 : [1.0 - 3.0] : "; cin >> alpha;
	cout << "베타값을 입력하시오 : [0 - 100] : "; cin >> beta;
	image.convertTo(oimage, 1, alpha, beta);
	imshow("Original Image", image);
	imshow("New Image", oimage);
	waitKey(0);
#endif
	//화소처리 - p.21
#if 0
	src = imread("D:\\999.Image\\lenna.jpg");
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	namedWindow("결과영상", WINDOW_AUTOSIZE);
	createTrackbar("임계값", "결과영상", &threshold_value, 255, Threshold_Demo);

	Threshold_Demo(0, 0);
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}
#endif
	//화소처리 - p.31
#if 0
	Mat src1, src2, dst;
	double gamma = 0.5;
	src1 = imread("D:\\999.Image\\gamma1.jpg");
	if (src1.empty())
	{
		cout << "영상을 읽을수 없습니다." << endl;
		return -1;
	}
	Mat table(1, 256, CV_8U);
	uchar* p = table.ptr();
	for (int i = 0; i < 256; i++)
	{
		p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	}

	LUT(src1, table, dst);
	imshow("src1", src1);
	imshow("dst", dst);
	waitKey(0);
#endif
	//히스토그램 - p.11
#if 0
	Mat	src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	imshow("Input Image", src);
	int	histogram[256] = { 0 };

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			histogram[(int)src.at<uchar>(y, x)]++;
		}
	}

	drawHist(histogram);
	waitKey(0);
#endif
	//히스토그램 - p.15
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg");
	if (src.empty())
	{
		return -1;
	}
	vector<Mat> bgr_planes;
	split(src, bgr_planes);
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumlate = false;

	Mat	b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumlate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumlate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumlate);
	//막대그래프가 그려지는 영상을 생성한다.
	int	hist_w = 512, hist_h = 400;
	int	bin_w = cvRound((double)hist_w / histSize);// 상자의 폭
	Mat	histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	//값들이 영상을 벗어나지 않도록 정규화한다.
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//	히스토그램의 값을 막대로 그린다.
	for (int i = 0; i < 255; i++)
	{
		line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - b_hist.at<float>(i)), Scalar(255, 0, 0));
		line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - g_hist.at<float>(i)), Scalar(0, 255, 0));
		line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - r_hist.at<float>(i)), Scalar(0, 0, 255));
	}

	imshow("입력 영상", src);
	imshow("컬러 히스토그램", histImage);
	waitKey();
#endif
	//히스토그램 - p.19
#if 0
	Mat image = imread("D:\\999.Image\\crayfish.jpg");
	Mat new_image = image.clone();

	int r1, r2, s1, s2; // r1 - ori min, r2 - ori max, s1 - stretch min, s2 - stretch max
	cout << "r1을 입력 하시오 : "; cin >> r1;
	cout << "r2을 입력 하시오 : "; cin >> r2;
	cout << "s1을 입력 하시오 : "; cin >> s1;
	cout << "s2을 입력 하시오 : "; cin >> s2;

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				int output = stretch(image.at<Vec3b>(y, x)[c], r1, s1, r2, s2);
				new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(output);
			}
		}
	}

	imshow("입력영상", image);
	imshow("출력영상", new_image);
	waitKey();
#endif
	//히스토그램 - p.24
#if 0
	Mat image = imread("D:\\999.Image\\equalize_test.jpg", 0);
	CV_Assert(!image.empty());

	Mat hist, dst1, dst2, hist_img, hist_img1, hist_img2;
	create_hist(image, hist, hist_img);

	//히스토그램 누적합 계산
	Mat accum_hist = Mat(hist.size(), hist.type(), Scalar(0));
	accum_hist.at<float>(0) = hist.at<float>(0);
	for (int i = 1; i < hist.rows; i++)
	{
		accum_hist.at<float>(i) = accum_hist.at<float>(i - 1) + hist.at<float>(i);
	}

	accum_hist /= sum(hist)[0];
	accum_hist *= 255;
	dst1 = Mat(image.size(), CV_8U);
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			int idx = image.at<uchar>(i, j);
			dst1.at<uchar>(i, j) = (uchar)accum_hist.at<float>(idx);
		}
	}

	equalizeHist(image, dst2);
	create_hist(dst1, hist, hist_img1);
	create_hist(dst2, hist, hist_img2);

	imshow("image", image);	imshow("img_hist", hist_img);
	imshow("dst1-User", dst1);	imshow("User-hist", hist_img1);
	imshow("dst2-OpenCV", dst2);	imshow("OpenCV_hist", hist_img2);
	waitKey();

#endif
	//공간필터링 p.8
#if 0
	Mat image = imread("D:\\999.Image\\filter_blur.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);

	float data[] = {
		1 / 9.f, 1 / 9.f , 1 / 9.f,
		1 / 9.f, 1 / 9.f , 1 / 9.f,
		1 / 9.f, 1 / 9.f , 1 / 9.f
	};

	Mat mask(3, 3, CV_32F, data);
	Mat blur;
	filter(image, blur, mask);
	blur.convertTo(blur, CV_8U);

	imshow("image", image);
	imshow("blur", blur);
	waitKey();
#endif
	//공간필터링 p.10
#if 0
	Mat image = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE); 

	float weight[] = {
		1 / 9.0f, 1 / 9.0f , 1 / 9.0f,
		1 / 9.0f, 1 / 9.0f , 1 / 9.0f,
		1 / 9.0f, 1 / 9.0f , 1 / 9.0f
	};
	Mat mask(3, 3, CV_32F, weight);
	Mat blur;
	filter2D(image, blur, -1, mask);
	blur.convertTo(blur, CV_8U);
	imshow("image", image);
	imshow("blur", blur);
	waitKey();
#endif
	//공간필터링 p.16
#if 0
	Mat image = imread("D:\\999.Image\\filter_sharpen.jpg", IMREAD_GRAYSCALE);

	CV_Assert(image.data);

	float data1[] = {
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0,
	};
	float data2[] = {
		-1, -1, -1,
		-1, 9, -1,
		-1, -1, -1,
	};

	Mat mask1(3, 3, CV_32F, data1);
	Mat mask2(3, 3, CV_32F, data2);

	Mat sharpen1, sharpen2;
	filter(image, sharpen1, mask1);
	filter(image, sharpen2, mask2);
	sharpen1.convertTo(sharpen1, CV_8U);
	sharpen2.convertTo(sharpen2, CV_8U);
	imshow("image", image);
	imshow("sharpen1", sharpen1);
	imshow("sharpen2", sharpen2);
	waitKey();
#endif
	//공간필터링 p.21
#if 0
	Mat src = imread("D:\\999.Image\\city1.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
	{
		return -1;
	}
	Mat dst;
	Mat noise_img = Mat::zeros(src.rows, src.cols, CV_8U);
	randu(noise_img, 0, 255);//noise_img의 모든화소를 0~255까지의 난수로 채움
	
	Mat black_img = noise_img < 10; // noise_img의 화소값이 10 보다 작으면 1이되는 black_img 생성
	Mat white_img = noise_img > 245; // noise_img의 화소값이 245 보다 크면 1이되는 white_img 생성

	Mat src1 = src.clone();
	src1.setTo(255, white_img); //white_img의 화소값이 1이면 src1의 화소값을 255로 한다. salt noise
	src1.setTo(0, black_img); //black_img의 화소값이 1이면 src1의 화소값을 0로 한다. pepper noise
	medianBlur(src1, dst, 5);
	imshow("srorce", src1);
	imshow("result", dst);
	waitKey();
#endif
	//공간필터링 p.27~28
#if 0
	Mat image = imread("D:\\999.Image\\edge_test1.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);

	//프르윗 마스크 원소
	float data1[] = { //수직마스크
		-1, 0, 1,
		-1, 0, 1,
		-1, 0, 1,
	};
	float data2[] = { //수평마스크
		-1, -1, -1,
		0, 0, 0,
		1, 1, 1,
	};

	Mat dst;
	differential(image, dst, data1, data2);
	imshow("image", image);
	imshow("프리윗 에지", dst);
	waitKey();
#endif
	//공간필터링 p.37
#if 0
	Mat src, src_gray, dst;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
	{
		return -1;
	}
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat abs_dst;
	Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	imshow("Image", src);
	imshow("Laplacian", abs_dst);
	waitKey();
#endif
	//공간필터링 p.44
#if 0
	src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
	{
		return -1;
	}
	dst.create(src.size(), src.type());
	namedWindow("Canny", WINDOW_AUTOSIZE);
	createTrackbar("Min Threshold:", "Canny", &lowThreshold, max_lowThreshold, CannyThreshold);
	CannyThreshold(0, 0);
	waitKey(0);

#endif
	//기하학적 변환 - p.10
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	Mat dst = Mat::zeros(Size(src.cols * 2, src.rows * 2), src.type());
	 
	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			float gx = ((float)x) / 2.0;
			float gy = ((float)y) / 2.0;
			int gxi = (int)gx;
			int gyi = (int)gy;
			float c00 = GetPixel(src, gxi, gyi);
			float c01 = GetPixel(src, gxi+1, gyi);
			float c10 = GetPixel(src, gxi, gyi+1);
			float c11 = GetPixel(src, gxi + 1, gyi+1);
			int value = (int)Blerp(c00, c10, c01, c11, gx - gxi, gy - gyi);
			dst.at<uchar>(y, x) = value;
		}
	}
	imshow("src", src);
	imshow("dst", dst);
	waitKey();
#endif
	//기하학적 변환 - p.18
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", IMREAD_COLOR);
	Point2f srcTri[3];
	Point2f dstTri[3];
	Mat warp_mat(2, 3, CV_32FC1);

	Mat warp_dst;
	warp_dst = Mat::zeros(src.rows, src.cols, src.type());
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.0f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.0f);
	dstTri[0] = Point2f(src.cols * 0.0f, src.rows * 0.33f);
	dstTri[1] = Point2f(src.cols * 0.85f, src.rows * 0.25f);
	dstTri[2] = Point2f(src.cols * 0.15f, src.rows * 0.7f);
	warp_mat = getAffineTransform(srcTri, dstTri);
	warpAffine(src, warp_dst, warp_mat, warp_dst.size());

	imshow("src", src);
	imshow("dst", warp_dst);
	waitKey();
#endif
	//기하학적 변환 - p.23
#if 0
	Mat src = imread("D:\\999.Image\\book.jpg");

	Point2f inputp[4];
	inputp[0] = Point2f(30, 81);
	inputp[1] = Point2f(274, 247);
	inputp[2] = Point2f(298, 40);
	inputp[3] = Point2f(598, 138);

	Point2f outputp[4];
	outputp[0] = Point2f(0, 0);
	outputp[1] = Point2f(0, src.rows);
	outputp[2] = Point2f(src.cols, 0);
	outputp[3] = Point2f(src.cols, src.rows);

	Mat h = getPerspectiveTransform(inputp, outputp);
	Mat out;
	warpPerspective(src, out, h, src.size());
	imshow("Source Image", src);
	imshow("Warped Source Image", out);
	waitKey();
#endif
	//12 - 형태학적 - 침식연산 - p.6
#if 0
	Mat image = imread("D:\\999.Image\\morph_test1.jpg", 0);
	CV_Assert(image.data);

	Mat th_img, dst1, dst2;
	threshold(image, th_img, 128, 255, THRESH_BINARY); //영상이진화

	uchar data[] = { 0,1,0,
					1,1,1,
					0,1,0 };
	Mat mask(3, 3, CV_8UC1, data); //마스크 선언 및 초기화

	erosion(th_img, dst1, (Mat)mask);	//사용자정의 침식 함수
	morphologyEx(th_img, dst2, MORPH_ERODE, mask); //OpenCV 침식 함수

	imshow("image", image), imshow("이진 영상", th_img);
	imshow("User_erosion", dst1);
	imshow("OpenCV_erosion", dst2);

	waitKey();
#endif
	//12 - 형태학적 - 팽창연산 - p.10
#if 0
	Mat image = imread("D:\\999.Image\\morph_test1.jpg", 0);
	CV_Assert(image.data);

	Mat th_img, dst1, dst2;
	threshold(image, th_img, 128, 255, THRESH_BINARY);

	Matx < uchar, 3, 3> mask;
	mask << 0, 1, 0,
		1, 1, 1,
		0, 1, 1;

	dilation(th_img, dst1, (Mat)mask);
	morphologyEx(th_img, dst2, MORPH_DILATE, mask);

	imshow("image", image);	
	imshow("User_dilation", dst1);
	imshow("OpenCV_dilation", dst2);
	waitKey();
#endif
	//12 - 형태학적 - 열림/닫힘연산 - p.18
#if 0
	Mat image = imread("D:\\999.Image\\morph_test1.jpg", 0);
	CV_Assert(image.data);
	Mat th_img, dst1, dst2, dst3, dst4;
	threshold(image, th_img, 128, 255, THRESH_BINARY);

	Matx < uchar, 3, 3> mask;
	mask << 0, 1, 0,
		1, 1, 1,
		0, 1, 0;

	opening(th_img, dst1, (Mat)mask);  //사용자 정의 함수 열림함수 호출
	closing(th_img, dst2, (Mat)mask);
	morphologyEx(th_img, dst3, MORPH_OPEN, mask); //OpenCV 열림함수
	morphologyEx(th_img, dst4, MORPH_CLOSE, mask);//OpenCV 닫힘함수

	imshow("User_opening", dst1);
	imshow("User_closing", dst2);
	imshow("OpenCV_opening", dst3);
	imshow("OpenCV_closing", dst4);
	waitKey();
#endif
	//12 - 형태학적 - 차량번호판 검출 - p.23
#if 0
	while (1)
	{
		int no;
		cout << "차량 영상 번호( 0:종료 ) : ";
		cin >> no;
		if (no == 0) break;

		string fname = format("D:\\999.Image\\test_car\\%02d.jpg", no);
		Mat image = imread(fname, 1);
		if (image.empty()) 
		{
			cout << to_string(no) + "번 영상 파일이 없습니다. " << endl;
			continue;
		}

		Mat gray, sobel, th_img, morph;
		Mat kernel(5, 25, CV_8UC1, Scalar(1));		// 닫힘 연산 마스크
		cvtColor(image, gray, COLOR_BGR2GRAY);		// 명암도 영상 변환

		blur(gray, gray, Size(5, 5));				// 블러링
		Sobel(gray, gray, CV_8U, 1, 0, 3);			// 소벨 에지 검출

		threshold(gray, th_img, 120, 255, THRESH_BINARY);	// 이진화 수행
		morphologyEx(th_img, morph, MORPH_CLOSE, kernel);	// 닫힘 연산 수행

		imshow("image", image);
		imshow("이진 영상", th_img);
		imshow("열림 연산", morph);
		waitKey();
	}
#endif
	//12 - 형태학적 - 골격화 - p.27
#if 0
	Mat img = imread("D:\\999.Image\\letterb.png", IMREAD_GRAYSCALE);
	threshold(img, img, 127, 255, THRESH_BINARY);

	imshow("src", img);
	Mat skel(img.size(), CV_8UC1, Scalar(0));
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));
	Mat temp, eroded;
	do
	{
		erode(img, eroded, element);
		dilate(eroded, temp, element);
		subtract(img, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(img);

	} while ((countNonZero(img) != 0));

	imshow("result", skel);
	waitKey();
#endif
	//13 - 컬러영상처리 - p.16
#if 0
	Mat BGR_img = imread("D:\\999.Image\\color_space.jpg", IMREAD_COLOR);
	CV_Assert(BGR_img.data);
	Mat HSI_img, HSV_img, hsi[3], hsv[3];

	bgr2hsi(BGR_img, HSI_img);
	cvtColor(BGR_img, HSV_img, COLOR_BGR2HSV);
	split(HSI_img, hsi);
	split(HSV_img, hsv);

	imshow("BGR_img", BGR_img);
	imshow("Hue", hsi[0]);
	imshow("Saturation", hsi[1]);
	imshow("Intensity", hsi[2]);
	imshow("OpenCV_Hue", hsv[0]);
	imshow("OpenCV_Saturation", hsv[1]);
	imshow("OpenCV_Value", hsv[2]);
	waitKey();
#endif

	//13 - 컬러영상처리 - p.21
#if 0
	Mat img = imread("D:\\999.Image\\image1.jpg", IMREAD_COLOR);
	if (img.empty()) 
	{ 
		return -1; 
	}

	Mat imgHSV;
	cvtColor(img, imgHSV, COLOR_BGR2HSV);

	Mat imgThresholded;
	inRange(imgHSV, Scalar(100, 0, 0), Scalar(120, 255, 255), imgThresholded);

	imshow("Thresholded Image", imgThresholded);
	imshow("Original", img);

	waitKey(0);
#endif

	//13 - 컬러영상처리 - p.22
#if 0
	VideoCapture cap("D:\\999.Image\\tennis_ball.mp4");
	if (!cap.isOpened())
	{
		return -1;
	}		
	for (;;)
	{
		Mat imgHSV;
		Mat frame;
		cap >> frame;
		cvtColor(frame, imgHSV, COLOR_BGR2HSV);

		Mat imgThresholded;
		inRange(imgHSV, Scalar(30, 10, 10), Scalar(38, 255, 255), imgThresholded);

		imshow("frame", frame);
		imshow("dst", imgThresholded);
		if (waitKey(30) >= 0) break;
	}
	waitKey(0);
#endif

	//13 - 컬러영상처리 - p.27
#if 0
	Mat src = imread("D:\\999.Image\\pepper.bmp", IMREAD_COLOR);
	if (src.empty())
	{
		cerr << "Image Load Failed!" << endl;
	}
	Mat src_ycrcb;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);

	vector<Mat> ycrcb_planes;
	split(src_ycrcb, ycrcb_planes);
	
	equalizeHist(ycrcb_planes[0], ycrcb_planes[0]);
	
	Mat dst_ycrcb;
	merge(ycrcb_planes, dst_ycrcb);

	Mat dst;
	cvtColor(dst_ycrcb, dst, COLOR_YCrCb2BGR);

	imshow("src", src);
	imshow("dst", dst);
	waitKey();
#endif
	//14 - 주파수영역처리 - p.10
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	Mat src_float;

	// 그레이스케일 영상을 실수 영상으로 변환한다.
	src.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	Mat dft_image;
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);	
	shuffleDFT(dft_image);
	displayDFT(dft_image);
#endif

	//14 - 주파수영역처리 - p.12
#if 0
	Mat img = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);

	Mat img_float, dft1, inversedft, inversedft1;
	img.convertTo(img_float, CV_32F);
	dft(img_float, dft1, DFT_COMPLEX_OUTPUT);

	// 역변환을 수행한다. 
	idft(dft1, inversedft, DFT_SCALE | DFT_REAL_OUTPUT);

	inversedft.convertTo(inversedft1, CV_8U);
	imshow("invertedfft", inversedft1);

	imshow("original", img);
	waitKey();
#endif

	//14 - 주파수영역처리 - p.17,21
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	Mat src_float;
	imshow("original", src);

	// 그레이스케일 영상을 실수 영상으로 변환한다.
	src.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	Mat dft_image;
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);
	shuffleDFT(dft_image);


	//p.17
	Mat highpass = getFilter_Circle(dft_image.size());
	//p.21
	//Mat highpass = getFilter(dft_image.size());
	Mat result;

	// 원형 필터와 DFT 영상을 서로 곱한다.
	multiply(dft_image, highpass, result);
	displayDFT(result);

	Mat inverted_image;
	shuffleDFT(result);
	idft(result, inverted_image, DFT_SCALE | DFT_REAL_OUTPUT);
	imshow("inverted", inverted_image);
	waitKey();
#endif
	//14 - 주파수영역처리 - p.26
#if 0
	Mat src = imread("D:\\999.Image\\lunar.png", IMREAD_GRAYSCALE);
	Mat src_float, dft_image;
	imshow("original", src);

	// 그레이스케일 영상을 실수 영상으로 변환한다.
	src.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);
	shuffleDFT(dft_image);
	displayDFT(dft_image);

	Mat lowpass = getFilter_Pattern(dft_image.size());
	Mat result;

	// 필터와 DFT 영상을 서로 곱한다.
	multiply(dft_image, lowpass, result);
	displayDFT(result);

	Mat inverted_image;
	shuffleDFT(result);
	idft(result, inverted_image, DFT_SCALE | DFT_REAL_OUTPUT);
	imshow("inverted", inverted_image);
	waitKey();
#endif

	//15 - 영상 분할 - p.6 ~ 7
#if 0
	src_Image_Seg = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	
	namedWindow("result", WINDOW_NORMAL);
	createTrackbar("임계값",	"result", &threshold_value_Image_Seg,	max_value_Image_Seg, MyThreshold);
	MyThreshold(0, 0); // 초기화를 위하여 호출한다. 
	waitKey();
#endif

	//15 - 영상 분할 - p.13
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	Mat blur, th1, th2, th3, th4;
	threshold(src, th1, 127, 255, THRESH_BINARY);
	threshold(src, th2, 0, 255, THRESH_BINARY | THRESH_OTSU);

	Size size = Size(5, 5);
	GaussianBlur(src, blur, size, 0);
	threshold(blur, th3, 0, 255, THRESH_BINARY | THRESH_OTSU);

	imshow("Original", src);
	imshow("Global", th1);
	imshow("Ostu", th2);
	imshow("Ostu after Blurring", th3);
	waitKey();
#endif
	//15 - 영상 분할 - p.19
#if 0
	Mat src = imread("D:\\999.Image\\book1.jpg", IMREAD_GRAYSCALE);
	Mat img, th1, th2, th3, th4;
	medianBlur(src, img, 5);
	threshold(img, th1, 127, 255, THRESH_BINARY);
	adaptiveThreshold(img, th2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
	adaptiveThreshold(img, th3, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

	imshow("Original", src);
	imshow("Global Thresholding", th1);
	imshow("Adaptive Mean", th2);
	imshow("Adaptive Gaussian", th3);
	waitKey();
#endif
	//15 - 영상 분할 - p.28
#if 0
	Mat img, img_edge, labels, centroids, img_color, stats;
	img = imread("D:\\999.Image\\coins.png", IMREAD_GRAYSCALE);

	threshold(img, img_edge, 128, 255, THRESH_BINARY_INV);
	imshow("Image after threshold", img_edge);

	int n = connectedComponentsWithStats(img_edge, labels, stats, centroids);

	vector<Vec3b> colors(n + 1);
	colors[0] = Vec3b(0, 0, 0);
	for (int i = 1; i <= n; i++) 
	{
		colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
	}
	img_color = Mat::zeros(img.size(), CV_8UC3);
	for (int y = 0; y < img_color.rows; y++)
	{
		for (int x = 0; x < img_color.cols; x++)
		{
			int label = labels.at<int>(y, x);
			img_color.at<Vec3b>(y, x) = colors[label];
		}
	}		

	imshow("Labeled map", img_color);
	waitKey();
#endif

	//15 - 영상 분할 - p.34
#if 0
	int coin_no = 20;
	String  fname = format("D:\\999.Image\\Coin\\%2d.png", coin_no);
	Mat  image = imread(fname, 1);
	CV_Assert(image.data);

	Mat th_img = preprocessing(image);
	vector<RotatedRect> circles = find_coins(th_img);

	for (int i = 0; i < circles.size(); i++)
	{
		float radius = circles[i].angle;
		circle(image, circles[i].center, radius, Scalar(0, 255, 0), 2);
	}

	imshow("전처리영상", th_img);
	imshow("동전영상", image);
	waitKey();
#endif

	//15 - 영상 분할 - p.39
#if 0
	Mat img = imread("D:\\999.Image\\polygon.bmp", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}

	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	Mat bin;
	threshold(gray, bin, 200, 255, THRESH_BINARY_INV | THRESH_OTSU);
	
	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
	
	for (vector<Point>& pts : contours)
	{
		if (contourArea(pts) < 400)
		{
			continue;
		}

		vector<Point> approx;
		approxPolyDP(pts, approx, arcLength(pts, true) * 0.02, true);

		int vtc = (int)approx.size();
		if (vtc == 3)
		{
			setLabel(img, pts, "TRI");
		}
		else if (vtc == 4)
		{
			setLabel(img, pts, "RECT");
		}
		else if (vtc > 4)
		{
			double len = arcLength(pts, true);
			double area = contourArea(pts);
			double ratio = 4. * CV_PI * area / (len * len);
			if (ratio > 0.8)
			{
				setLabel(img, pts, "CIR");
			}
		}
	}

	imshow("img", img);
	waitKey();
#endif
	//16 - 특징추출 - p.22
#if 0
	Mat src = imread("D:\\999.Image\\building.jpg", 0);
	if (src.empty()) 
	{
		cout << "can not open " << endl;    
		return -1; 
	}

	Mat dst, cdst;
	Canny(src, dst, 100, 200);
	imshow("edge", dst);
	cvtColor(dst, cdst, COLOR_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 100, 20);
	for (size_t i = 0; i < lines.size(); i++) 
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}

	imshow("source", src);
	imshow("detected lines", cdst);
	waitKey();
#endif
	//16 - 특징추출 - p.27
#if 0
	Mat src, src_gray;

	src = imread("D:\\999.Image\\plates.jpg", 1);
	imshow("src", src);
	// 그레이스케일로 변환한다. 
	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	// 가우시안 블러링 적용
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	vector<Vec3f> circles;

	// 원을 검출하는 허프 변환
	HoughCircles(src_gray, circles, HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 50, 0, 0);

	// 원을 영상 위에 그린다. 
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0); // 원의 중심을 그린다. 
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0); // 원을 그린다.
	}

	imshow("Hough Circle Transform", src);
	waitKey();
#endif

	//17 - 특징추출(2) - p.12 ~ 14
#if 0
	image = imread("D:\\999.Image\\harris_test.jpg", 1);			// 컬러 영상입력
	CV_Assert(image.data);

	int blockSize = 4;
	int apertureSize = 3;
	double k = 0.04;
	int  thresh = 20;
	Mat gray;

	cvtColor(image, gray, COLOR_BGR2GRAY);
	cornerharris(gray, corner1, blockSize, apertureSize, k); 	// 직접 구현 함수
	cornerHarris(gray, corner2, blockSize, apertureSize, k);	// OpenCV 제공 함수

	cornerHarris_demo(0, 0);
	createTrackbar("Threshold: ", "img1-User harris", &thresh, 100, cornerHarris_demo);
	waitKey();
#endif

	//17 - 특징추출(2) - p.19
#if 0
	corner_fast();
#endif

	//17 - 특징추출(2) - p.37
#if 0
	detect_kepoints();
#endif

	//18 - 특징매칭 - p.12
#if 0
	template_matching();
#endif

	//18 - 특징매칭 - p.29
#if 0
	keypoint_matching();
#endif

	//18 - 특징매칭 - p.39 ~ 41
#if 0
	find_homography();
#endif
	//19 - 영상분류 - p.9 ~ 10
#if 0
	Mat train_features(5, 2, CV_32FC1);
	Mat labels(5, 1, CV_32FC1);

	// 점의 좌표를 train_features에 입력한다.  
	train_features.at<float>(0, 0) = 10, train_features.at<float>(0, 1) = 10;
	train_features.at<float>(1, 0) = 10, train_features.at<float>(1, 1) = 20;
	train_features.at<float>(2, 0) = 20, train_features.at<float>(2, 1) = 10;
	train_features.at<float>(3, 0) = 30, train_features.at<float>(3, 1) = 30;
	train_features.at<float>(4, 0) = 40, train_features.at<float>(4, 1) = 30;

	// 원하는 레이블을 labels에 입력한다. 
	labels.at<float>(0, 0) = 1;
	labels.at<float>(1, 0) = 1;
	labels.at<float>(2, 0) = 1;
	labels.at<float>(3, 0) = 2;
	labels.at<float>(4, 0) = 2;

	// 학습 과정
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
	knn->train(trainData);

	// 테스트 과정
	Mat sample(1, 2, CV_32FC1);
	Mat predictedLabels;

	// 테스트 데이터를 입력한다. 
	sample.at<float>(0, 0) = 28, sample.at<float>(0, 1) = 28;
	knn->findNearest(sample, 2, predictedLabels);

	float prediction = predictedLabels.at<float>(0, 0);
	cout << "테스트 샘플의 라벨 = " << prediction << endl;
	waitKey();
#endif

	//19 - 영상분류 - p.12 ~ 13
#if 1
	Mat img;
	img = imread("D:\\999.Image\\digits.png", IMREAD_GRAYSCALE);
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", img);
	waitKey(0);

	Mat train_features(5000, 400, CV_32FC1);
	Mat labels(5000, 1, CV_32FC1);

	// 각 숫자 영상을 행 벡터로 만들어서 train_feature에 저장한다. 
	for (int r = 0; r < 50; r++) 
	{
		for (int c = 0; c < 100; c++) 
		{
			int i = 0;
			for (int y = 0; y < 20; y++) 
			{
				for (int x = 0; x < 20; x++) 
				{
					train_features.at<float>(r * 100 + c, i++) = img.at<uchar>(r * 20 + y, c * 20 + x);
				}
			}
		}
	}

	// 각 숫자 영상에 대한 레이블을 저장한다. 
	for (int i = 0; i < 5000; i++) 
	{
		labels.at<float>(i, 0) = (i / 500);
	}

	// 학습 과정
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	Ptr<ml::TrainData> trainData = ml::TrainData::create(train_features, ml::ROW_SAMPLE, labels);
	knn->train(trainData);

	// 테스트 과정
	Mat predictedLabels;
	for (int i = 0; i < 5000; i++) 
	{
		Mat test = train_features.row(i);
		knn->findNearest(test, 3, predictedLabels);
		float prediction = predictedLabels.at<float>(0);
		cout << "테스트 샘플" << i << "의 라벨 = " << prediction << '\n';
	}
	waitKey();
#endif
	//19 - 영상분류 - p.19 ~ 20
#if 0
	Mat samples(50, 2, CV_32F);

	for (int y = 0; y < samples.rows; y++)
	{
		samples.at<float>(y, 0) = (rand() % 255);
		samples.at<float>(y, 1) = (rand() % 255);
	}
	Mat dst(256, 256, CV_8UC3);

	for (int y = 0; y < samples.rows; y++) 
	{
		float x1 = samples.at<float>(y, 0);
		float x2 = samples.at<float>(y, 1);
		circle(dst, Point(x1, x2), 3, Scalar(255, 0, 0));
	}
	imshow("dst", dst);

	Mat result;
	Mat labels(50, 1, CV_8UC1);

	Mat centers;
	result = Mat::zeros(Size(256, 256), CV_8UC3);
	kmeans(samples, 2, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
		3, KMEANS_PP_CENTERS, centers);

	for (int y = 0; y < samples.rows; y++) 
	{
		float x1 = samples.at<float>(y, 0);
		float x2 = samples.at<float>(y, 1);
		int cluster_idx = labels.at<int>(y, 0);
		if (cluster_idx == 0)
			circle(result, Point(x1, x2), 3, Scalar(255, 0, 0));
		else
			circle(result, Point(x1, x2), 3, Scalar(255, 255, 0));
	}
	imshow("result", result);
	waitKey(0);
#endif
	//19 - 영상분류 - p.21
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", 1);

	// 학습 데이터를 만든다. 
	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			for (int z = 0; z < 3; z++)
			{
				samples.at<float>(y + x * src.rows, z) = src.at<Vec3b>(y, x)[z];
			}				
		}			
	}

	// 클러스터의 개수는 15가 된다. 
	int clusterCount = 15;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001),
		attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * src.rows, 0);
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}		
	imshow("src", src);
	imshow("clustered image", new_image);
	waitKey(0);
#endif
	return 0;
}
