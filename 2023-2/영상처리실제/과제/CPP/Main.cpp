#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include <stdio.h>

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

typedef unsigned char BYTE;

Mat img_p26;
int nDrawing_p26 = false;

Mat img_p30;
string title = "Ʈ�����̺�Ʈ";

Mat img_p33;
int red, green, blue;
int nDrawing_p33 = false;
void on_trackbar(int, void*) {}

Mat img_hw1;
int nDrawing_hw1 = false;
void on_trackbar_hw1(int, void*) {}
int nLineThinkness = 0;
int nRadius = 0;


//ȭ��ó�� HW1
Mat img_Pixel_Processing;
int mx1, my1, mx2, my2;
bool cropping = false;

//ȭ��ó�� HW2
Mat img_Pixel_Processing_src1;
Mat img_Pixel_Processing_src2;
Mat img_Pixel_Processing_dst;
int nAlpha_Pixel_Processing = 0;
string title_Pixel_Processing_HW2 = "Ʈ�����̺�Ʈ";

//������׷�
Mat img_Histogram;


void onMouse_Histogram(int event, int x, int y, int flags, void* param)
{
#if 1
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������
		mx1 = x; // �簢���� ���� ��� ��ǥ ����
		my1 = y;
		cropping = true;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
	}
	else if (event == EVENT_LBUTTONUP)
	{
		// ���콺�� ���� ��ư���� ���� ����
		mx2 = x; // �簢���� ���� �ϴ� ��ǥ ����
		my2 = y;
		cropping = false;
		Mat dst(img_Histogram, (Rect(mx1, my1, mx2 - mx1, my2 - my1)));


		vector<Mat> bgr_planes;
		split(dst, bgr_planes);
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true, accumlate = false;

		Mat	b_hist, g_hist, r_hist;
		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumlate);
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumlate);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumlate);
		//����׷����� �׷����� ������ �����Ѵ�.
		int	hist_w = 512, hist_h = 400;
		int	bin_w = cvRound((double)hist_w / histSize);// ������ ��
		Mat	histImage_B(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		Mat	histImage_G(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		Mat	histImage_R(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		//������ ������ ����� �ʵ��� ����ȭ�Ѵ�.
		normalize(b_hist, b_hist, 0, histImage_B.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, histImage_G.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, histImage_R.rows, NORM_MINMAX, -1, Mat());

		//	������׷��� ���� ����� �׸���.
		for (int i = 0; i < 255; i++)
		{
			line(histImage_B, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - b_hist.at<float>(i)), Scalar(255, 0, 0));
			line(histImage_G, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - g_hist.at<float>(i)), Scalar(0, 255, 0));
			line(histImage_R, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - r_hist.at<float>(i)), Scalar(0, 0, 255));
		}
		imshow("�÷� ������׷� - Blue", histImage_B);
		imshow("�÷� ������׷� - Green", histImage_G);
		imshow("�÷� ������׷� - Red", histImage_R);
		
	}
#endif 
}

void onMouse_Pixel_Processing(int event, int x, int y, int flags, void* param)
{
	//ȭ��ó�� HW1
#if 1
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������
		mx1 = x; // �簢���� ���� ��� ��ǥ ����
		my1 = y;
		cropping = true;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
	}
	else if (event == EVENT_LBUTTONUP)
	{
		// ���콺�� ���� ��ư���� ���� ����
		mx2 = x; // �簢���� ���� �ϴ� ��ǥ ����
		my2 = y;
		cropping = false;
		//rectangle(img, Rect(mx1, my1, mx2 - mx1, my2 - my1), Scalar(0, 255, 0), 2);


		Mat dst(img_Pixel_Processing, (Rect(mx1, my1, mx2 - mx1, my2 - my1)));
		dst = 255 - dst;

		imshow("image", img_Pixel_Processing);
	}
#endif 

}

void on_trackbar_Pixel_Processing(int nAlpha, void* pUserdata)
{	
	double dTemp = ((double)nAlpha / 10);

	double beta = (1.0 - dTemp);

	addWeighted(img_Pixel_Processing_src1, dTemp, img_Pixel_Processing_src2, beta, 0.0, img_Pixel_Processing_dst);
	imshow(title_Pixel_Processing_HW2, img_Pixel_Processing_dst);
}

void onChange(int nValue, void* pUserdata)
{
	int nAdd_Value = nValue - 130;
	cout << "�߰� ȭ�Ұ�" << nAdd_Value << endl;

	Mat temp = img_p30 + nAdd_Value;
	imshow(title, temp);
}

void drawCircle(int event, int x, int y, int flags, void* param)
{
	//3�� 26page
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

	//3�� 33page
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

		imshow("hw1", img);// ������ ����Ǹ� �ٽ� ǥ���Ѵ�
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		Mat img = *(Mat*)(param);
		Point_<int> pt1(x, y);
		Rect rect(pt1, Size(100, 100));
		if (nLineThinkness <= 0)
		{
			rectangle(img, rect, blue, 2); //�簢�� �׸���
		}
		else
		{
			rectangle(img, rect, blue, nLineThinkness); //�簢�� �׸���
		}


		imshow("hw1", img);// ������ ����Ǹ� �ٽ� ǥ���Ѵ�
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
	//3�� 19page
#if 0
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		cout << "���콺 ���ʹ�ư ������" << endl;
		break;
	case EVENT_RBUTTONDOWN:
		cout << "���콺 �����ʹ�ư ������" << endl;
		break;
	case EVENT_LBUTTONUP:
		cout << "���콺 ���ʹ�ư ����" << endl;
		break;
	case EVENT_RBUTTONUP:
		cout << "���콺 �����ʹ�ư ����" << endl;
		break;
	}
#endif

	//3�� 23page
#if 0
	if (event == EVENT_LBUTTONDOWN)
	{
		Mat img = *(Mat*)(param);
		circle(img, Point(x, y), 200, Scalar(0, 255, 0), 10);
		putText(img, "I found a dog!", Point(x, y + 200), FONT_HERSHEY_PLAIN, 2.0, 255, 2);
		imshow("src ", img);// ������ ����Ǹ� �ٽ� ǥ���Ѵ�
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

void medianFilter(Mat input, Mat& output, int ksize)
{
	vector<uchar> neighbors;
	uchar sample = 0;
	uchar median = 0;

	int nType = output.type();

	for (int y = 0; y < output.rows; y++) 
	{
		for (int x = 0; x < output.cols; x++)
		{
			for (int s = 0; s < ksize; s++)
			{
				for (int t = 0; t < ksize; t++)
				{

					//padding	
					sample = input.at<uchar>(min(output.rows - 1, max(0, y + t)), min(output.cols - 1, max(0, x + s)));
					neighbors.push_back(sample);
				}
			}
		//find median value >dst(y,x)����
		sort(neighbors.begin(), neighbors.end());
		median = neighbors[neighbors.size() / 2];
		output.at<uchar>(y, x) = median;
		neighbors.clear();
		}
	}
}

Mat _11_HW1_src;
Mat warp_mat(2, 3, CV_32FC1);
Mat warp_dst;
int nMouseClickCount_11_HW1_src = 0;
int nMouseClickCount_11_HW1_dst = 0;
Point2f nMousePt_11_HW1_src[3];
Point2f nMousePt_11_HW1_dst[3];

void onMouse_11_HW1_src(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������		
		switch (nMouseClickCount_11_HW1_src)
		{
		case 0:
			nMousePt_11_HW1_src[0].x = x;
			nMousePt_11_HW1_src[0].y = y;
			nMouseClickCount_11_HW1_src++;
			break;
		case 1:
			nMousePt_11_HW1_src[1].x = x;
			nMousePt_11_HW1_src[1].y = y;
			nMouseClickCount_11_HW1_src++;
			break;
		case 2:
			nMousePt_11_HW1_src[2].x = x;
			nMousePt_11_HW1_src[2].y = y;
			nMouseClickCount_11_HW1_src = 0;
			break;
		}
		
	}
}
void onMouse_11_HW1_dst(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������		
		switch (nMouseClickCount_11_HW1_dst)
		{
		case 0:
			nMousePt_11_HW1_dst[0].x = x;
			nMousePt_11_HW1_dst[0].y = y;
			nMouseClickCount_11_HW1_dst++;
			break;
		case 1:
			nMousePt_11_HW1_dst[1].x = x;
			nMousePt_11_HW1_dst[1].y = y;
			nMouseClickCount_11_HW1_dst++;
			break;
		case 2:
			nMousePt_11_HW1_dst[2].x = x;
			nMousePt_11_HW1_dst[2].y = y;
			nMouseClickCount_11_HW1_dst = 0;
			
			warp_mat = getAffineTransform(nMousePt_11_HW1_src, nMousePt_11_HW1_dst);
			warpAffine(_11_HW1_src, warp_dst, warp_mat, warp_dst.size());
			imshow("11-HW1-DST", warp_dst);
			waitKey();

			break;
		}

	}
}


Mat _11_HW2_src;
Mat perspective_Transform;
Mat perspective_dst;
int nMouseClickCount_11_HW2_src = 0;
int nMouseClickCount_11_HW2_dst = 0;
Point2f nMousePt_11_HW2_src[4];
Point2f nMousePt_11_HW2_dst[4];
void onMouse_11_HW2_src(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������		
		switch (nMouseClickCount_11_HW2_src)
		{
		case 0:
			nMousePt_11_HW2_src[0].x = x;
			nMousePt_11_HW2_src[0].y = y;
			nMouseClickCount_11_HW2_src++;

			printf("ori - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_src[0].x, nMousePt_11_HW2_src[0].y, nMouseClickCount_11_HW2_src);
			break;
		case 1:
			nMousePt_11_HW2_src[1].x = x;
			nMousePt_11_HW2_src[1].y = y;
			nMouseClickCount_11_HW2_src++;
			printf("ori - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_src[1].x, nMousePt_11_HW2_src[1].y, nMouseClickCount_11_HW2_src);
			break;
		case 2:
			nMousePt_11_HW2_src[2].x = x;
			nMousePt_11_HW2_src[2].y = y;
			nMouseClickCount_11_HW2_src++;
			printf("ori - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_src[2].x, nMousePt_11_HW2_src[2].y, nMouseClickCount_11_HW2_src);
			break;
		case 3:
			nMousePt_11_HW2_src[3].x = x;
			nMousePt_11_HW2_src[3].y = y;
			nMouseClickCount_11_HW2_src = 0;
			printf("ori - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_src[3].x, nMousePt_11_HW2_src[3].y, nMouseClickCount_11_HW2_src);
			break;
		}

	}
}
void onMouse_11_HW2_dst(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		// ���콺�� ���� ��ư�� ������		
		switch (nMouseClickCount_11_HW2_dst)
		{
		case 0:
			nMousePt_11_HW2_dst[0].x = x;
			nMousePt_11_HW2_dst[0].y = y;
			nMouseClickCount_11_HW2_dst++;

			printf("dst - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_dst[0].x, nMousePt_11_HW2_dst[0].y, nMouseClickCount_11_HW2_dst);
			break;
		case 1:
			nMousePt_11_HW2_dst[1].x = x;
			nMousePt_11_HW2_dst[1].y = y;
			nMouseClickCount_11_HW2_dst++;
			printf("dst - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_dst[1].x, nMousePt_11_HW2_dst[1].y, nMouseClickCount_11_HW2_dst);
			break;
		case 2:
			nMousePt_11_HW2_dst[2].x = x;
			nMousePt_11_HW2_dst[2].y = y;
			nMouseClickCount_11_HW2_dst++;
			printf("dst - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_dst[2].x, nMousePt_11_HW2_dst[2].y, nMouseClickCount_11_HW2_dst);
			break;
		case 3:
			nMousePt_11_HW2_dst[3].x = x;
			nMousePt_11_HW2_dst[3].y = y;
			nMouseClickCount_11_HW2_dst = 0;
			printf("dst - X : %f, Y : %f, Count : %d\n", nMousePt_11_HW2_dst[3].x, nMousePt_11_HW2_dst[3].y, nMouseClickCount_11_HW2_dst);

			perspective_Transform = getPerspectiveTransform(nMousePt_11_HW2_src, nMousePt_11_HW2_dst);
			warpPerspective(_11_HW2_src, perspective_dst, perspective_Transform, _11_HW2_src.size());
			imshow("11-HW2-DST", perspective_dst);
			waitKey();

			break;
		}

	}
}

// ���� ���� ����
Point2f  pts[4], small(10, 10);							// 4�� ��ǥ �� ��ǥ �簢�� ũ��
Mat image;												// �Է� ���� 

void draw_rect(Mat image)								// 4�� ��ǥ �մ� �簢�� �׸���
{
	Rect img_rect(Point(0, 0), image.size());			// �Է¿��� ũ�� �簢��
	for (int i = 0; i < 4; i++)
	{
		Rect rect(pts[i] - small, pts[i] + small);		// ��ǥ �簢��
		rect &= img_rect;								// ���� ���� ���
		image(rect) += Scalar(70, 70, 70);				// �簢�� ���� ��� �ϱ�
		line(image, pts[i], pts[(i + 1) % 4], Scalar(255, 0, 255), 1);
		rectangle(image, rect, Scalar(255, 255, 0), 1);	// ��ǥ �簢�� �׸���
	}
	imshow("select rect", image);
}

void warp(Mat image)									// ���� ��ȯ ���� �Լ�
{
	Point2f dst_pts[4] = {								// ���� ���� 4�� ��ǥ
		Point2f(0, 0), Point2f(350, 0),
		Point2f(350, 350), Point2f(0, 350)
	};
	Mat dst;
	Mat perspect_mat = getPerspectiveTransform(pts, dst_pts);		// ���ٺ�ȯ ��� ���
	warpPerspective(image, dst, perspect_mat, Size(350, 350), INTER_CUBIC);
	imshow("�ְ��", dst);
}

void  onMouse_Test(int event, int x, int y, int flags, void*)	// ���콺 �̺�Ʈ ����
{
	Point curr_pt(x, y);									// ���� Ŭ�� ��ǥ
	static int check = -1;								// ���콺 ���� ��ǥ��ȣ

	if (event == EVENT_LBUTTONDOWN) {		// ���콺 �� ��ư 
		for (int i = 0; i < 4; i++)
		{
			Rect rect(pts[i] - small, pts[i] + small);	// ��ǥ �簢���� ����
			if (rect.contains(curr_pt))  check = i;		// ���� ��ǥ �簢�� ã��
		}
	}
	if (event == EVENT_LBUTTONUP)
		check = -1;									// ���� ��ǥ��ȣ �ʱ�ȭ

	if (check >= 0) {									// ��ǥ �簢�� ���ý�	
		pts[check] = curr_pt;							// Ŭ�� ��ǥ�� ���� ��ǥ�� ����
		draw_rect(image.clone());						// 4�� ��ǥ ���� �簢�� �׸���
		warp(image.clone());							// ���� ��ȯ ����
	}
}


//8�� - �������� ����
//����ũ ���ҿ� ����ũ ���� �Է�ȭ�� ���� ��ġ ���� üũ
bool check_match(Mat img, Point start, Mat mask, int mode = 0)
{
	for (int u = 0; u < mask.rows; u++)
	{
		for (int v = 0; v < mask.cols; v++)
		{
			Point pt(v, u);						//��ȸ��ǥ
			int m = mask.at<uchar>(pt);			//����ũ ���
			int p = img.at<uchar>(start + pt);	//�ش� ��ġ �Է�ȭ�� 

			bool ch = (p == 255);				//��ġ ���� ��
			if (m == 1 && ch == mode)			//mode 0�̸� ħ��, 1�̸� ��â
			{
				return  false;
			}
		}
	}
	return true;
}

//ħ�� ����
void erosion(Mat img, Mat& dst, Mat mask)
{
	dst = Mat(img.size(), CV_8U, Scalar(0));
	if (mask.empty())	mask = Mat(3, 3, CV_8UC1, Scalar(0));

	Point h_m = mask.size() / 2; //����ũ ���� ũ��

	for (int i = h_m.y; i < img.rows - h_m.y; i++)
	{
		for (int j = h_m.x; j < img.cols - h_m.x; j++)
		{
			Point start = Point(j, i) - h_m;
			bool  check = check_match(img, start, mask, 0); //���� ��ġ���� ��
			dst.at<uchar>(i, j) = (check) ? 255 : 0;		// ���ȭ�� ����
		}
	}
}

//��â����
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
			bool  check = check_match(img, start, mask, 1);	// ���� ��ġ���� ��
			dst.at<uchar>(i, j) = (check) ? 0 : 255;			// ħ�Ŀ���� �ݴ�
		}
	}
}

//��������
void opening(Mat img, Mat& dst, Mat mask)
{
	Mat tmp;
	erosion(img, tmp, mask);
	dilation(tmp, dst, mask);
}

//��������
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


Mat img_9_HW, roi_9_HW;
int mx1_9_HW, my1_9_HW, mx2_9_HW, my2_9_HW;

void onMouse_9_Color_Processing(int event, int x, int y, int flags, void* param)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		mx1_9_HW = x;
		my1_9_HW = y;
	}
	else if (event == EVENT_LBUTTONUP)
	{
		mx2_9_HW = x;
		my2_9_HW = y;

		if (mx1_9_HW <= mx2_9_HW && my1_9_HW <= my2_9_HW)
		{
			roi_9_HW = img_9_HW(Rect(mx1_9_HW, my1_9_HW, mx2_9_HW - mx1_9_HW, my2_9_HW - my1_9_HW));
		}
		else if (mx1_9_HW > mx2_9_HW && my1_9_HW <= my2_9_HW)
		{
			roi_9_HW = img_9_HW(Rect(mx2_9_HW, my1_9_HW, mx1_9_HW - mx2_9_HW, my2_9_HW - my1_9_HW));
		}
		else if(mx1_9_HW <= mx2_9_HW && my1_9_HW > my2_9_HW)
		{
			roi_9_HW = img_9_HW(Rect(mx1_9_HW, my2_9_HW, mx2_9_HW - mx1_9_HW, my1_9_HW - my2_9_HW));
		}
		else
		{
			roi_9_HW = img_9_HW(Rect(mx2_9_HW, my2_9_HW, mx1_9_HW - mx2_9_HW, my1_9_HW - my2_9_HW));
		}	
		
		imshow("ROI", roi_9_HW);

		Mat img_HSV;
		cvtColor(roi_9_HW, img_HSV, COLOR_BGR2HSV);

		Mat arrayHSV[3];
		split(img_HSV, arrayHSV);
		imshow("Hue", arrayHSV[0]);

		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };		

		Mat	Hue_hist;
		calcHist(&arrayHSV[0], 1, 0, Mat(), Hue_hist, 1, &histSize, &histRange);

		int	hist_w = 512, hist_h = 400;
		int	bin_w = cvRound((double)hist_w / histSize);// ������ ��
		Mat	histImage(hist_h, hist_w, CV_8UC3, Scalar(255,255,255));
		normalize(Hue_hist, Hue_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		for (int i = 0; i < 255; i++)
		{
			line(histImage, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - Hue_hist.at<float>(i)), Scalar(0, 0, 0));
		}
		imshow("Hue Histogram", histImage);
	}
}

void displayDFT(Mat& src, String strTitle)
{
	Mat image_array[2] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	// �� DFT ��� ������ 2���� �������� �и��Ѵ�. 
	split(src, image_array);

	Mat mag_image;
	// �� Ǫ���� ��ȯ ������� ���밪�� ����Ѵ�. 
	magnitude(image_array[0], image_array[1], mag_image);

	// �� Ǫ���� ��ȯ ������� ����� ũ�� ������ �α� �����Ϸ� ��ȯ�Ѵ�. 
	// 0���� ������ �ʵ��� 1�� �����ش�. 
	mag_image += Scalar::all(1);
	log(mag_image, mag_image);

	// �� 0���� 255�� ������ ����ȭ�Ѵ�. 
	normalize(mag_image, mag_image, 0, 1, NORM_MINMAX);
	imshow(strTitle, mag_image);
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

// ���� ���͸� �����.(������)
Mat getFilter_Circle(Size size)
{
	Mat filter(size, CV_32FC2, Vec2f(0, 0));
	circle(filter, size / 2, 50, Vec2f(1, 1), -1);
	return filter;
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
	//cout << "�ڳ� ����: " << cnt << endl;
	return image;
}

void template_matching()
{
	Mat img = imread("D:\\999.Image\\pcb.jpg", IMREAD_COLOR);

	Mat templ = imread("D:\\999.Image\\pcb_temp.jpg", IMREAD_COLOR);

	if (img.empty() || templ.empty())
	{
		cerr << " Image load failed!" << endl;
		return;
	}

	Mat res, res_norm;
	double maxv;
	Point maxloc;
	matchTemplate(img, templ, res, TM_CCOEFF_NORMED);
	normalize(res, res_norm, 0, 255, NORM_MINMAX, CV_8U);

	minMaxLoc(res, 0, &maxv, 0, &maxloc);

	for (int y = 0; y < res.rows; y++)
	{
		for (int x = 0; x < res.cols; x++)
		{
			if (res.at<float>(y, x) > 0.80f)
			{
				rectangle(img, Point(x, y), Point(x + templ.cols, y + templ.rows), Scalar(0, 0, 255), 2);
			}
		}
	}

	imshow("templ", templ);
	resize(img, img, Size(1024, 768));
	imshow("img", img);
	waitKey();
	destroyAllWindows();
}
void Keypoint_Matching()
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "ī�޶� ���� ���� �ʾҽ��ϴ�." << endl;
		exit(1);
	}

	double fps = 15;
	int delay = cvRound(1000.0 / fps);	
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');

	VideoWriter writer;//���������� ���� ��ü

	for (;;)
	{
		Mat frame;

		capture.read(frame);
#if 1
		Mat src1 = imread("D:\\999.Image\\HW_templ.png", IMREAD_GRAYSCALE); 
		Mat src2 = frame.clone();

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

		bool bOpened = writer.isOpened();

		if(!bOpened)
		{
			Size size(dst.cols, dst.rows);
			writer.open("D:\\HW_Video.avi", fourcc, fps, size);
		}		
		writer.write(dst);

		imshow("dst", dst);
		
#endif
		if (waitKey(delay) >= 0)
		{
			break;
		}
	}
}




int main()
{
	//1�� HW
#if 0
	Mat Image = imread("D:\\1.��������\\2.����ΰ������а�\\2.23��2�б�(����2�б�)\\2.����ó������\\2.����\\lenna.jpg", IMREAD_COLOR);

	if (Image.empty())
	{
		cout << "������ ���� �� ����" << endl;
	}

	imshow("src", Image);

	Mat gray, edge, output;
	cvtColor(Image, gray, COLOR_BGR2GRAY);

	imshow("gray", gray);

	imwrite("D:\\gray.jpg", gray);

	waitKey(0);
#endif
	//2��
	//8 Page
#if 0
	//Point_ ��ü ���� ���
	Point_<int> pt1(100, 200);
	Point_<float> pt2(92.3f, 125.23f);
	Point_<double> pt3(100.2, 300.9);

	//Point_ ��ü ���� ���� ���
	Point2i pt4(120, 69);
	Point2f pt5(0.3f, 0.f), pt6(0.f, 0.4f);
	Point2d pt7(0.25, 0.6);

	//Pint_ ��ü����
	Point pt8 = pt1 + (Point)pt2; //�ڷ����� �ٸ� Point ��ü ����
	Point2f pt9 = pt6 * 3.14f;
	Point2d pt10 = (pt3 + (Point2d)pt6) * 10;
	cout << "pt8 = " << pt8.x << " , " << pt8.y << endl;
	cout << "[pt9] = " << pt9 << endl;
	cout << "(pt2 == pt6)" << endl;
	cout << "pt7�� pt8�� ���� :" << pt7.dot(pt8) << endl;
#endif

	//13 Page
#if 0
	//Size_ ��ü ���� ���
	Size_<int> sz1(100, 200);
	Size_<float> sz2(192.3f, 25.3f);
	Size_<double> sz3(100.2, 30.9);

	//Size_ ��ü ���� ���� ���6
	Size sz4(120, 69);
	Size2f sz5(0.3f, 0.f);
	Size2d sz6(0.25, 0.6);

	Point2d pt1(0.25, 0.6);

	Size2i sz7 = sz1 + (Size2i)sz2;
	Size2d sz8 = sz3 - (Size2d)sz4;
	Size2d sz9 = sz5 + (Size2f)pt1;

	cout << "sz1.width = " << sz1.width;
	cout << ", sz1.height = " << sz1.height << endl;
	cout << "sz1 ���� :  " << sz1.area() << endl;
	cout << "[sz7] = " << sz7 << endl;
	cout << "[sz8] = " << sz8 << endl;
	cout << "[sz9] = " << sz9 << endl;

#endif

	//19 Page
#if 0
	Size2d sz(100.5, 60.6);
	Point2f pt1(20.f, 30.f), pt2(100.f, 200.f);

	//Rect_ ��ü �⺻ ���� ��� 
	Rect_<int> rect1(10, 10, 30, 50);
	Rect_<float> rect2(pt1, pt2);
	Rect_<double> rect3(Point2d(20.5, 10), sz);

	//Rect ���� ���� ��� & ���� ����
	Rect rect4 = rect1 + (Point)pt1;
	Rect2f rect5 = rect2 + (Size2f)sz;
	Rect2d rect6 = rect1 & (Rect)rect2;

	//��� 
	cout << "rect3 = " << rect3.x << ", " << rect3.y << ", ";
	cout << rect3.width << " x " << rect3.height << endl;
	cout << "rect4 = " << rect4.tl() << " " << rect4.br() << endl;
	cout << "rect5 ũ�� = " << rect5.size() << endl;
	cout << "[rect6]  = " << rect6 << endl;
#endif

	//21 Page
#if 0
	//�⺻���� �� ���� ���
	Vec <int, 2> v1(5, 12);
	Vec <double, 3> v2(40, 130.7, 125.6);
	Vec2d v3(10, 10);
	Vec6f v4(40.f, 230.25f, 525.6f);
	Vec3i v5(200, 230, 250);

	//��ü ���� �� ����ȯ
	Vec3d v6 = v2 + (Vec3d)v5;
	Vec2d v7 = (Vec2d)v1 + v3;
	Vec6f v8 = v4 * 20.0f;

	Point pt1 = v1 + (Vec2i)v7;
	Point3_<int> pt2 = static_cast<Point3_<int>>(v2);

	//�ܼ�â ���
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
	Scalar blue(255, 0, 0), red(0, 0, 255), green(0, 255, 0); //���󼱾�
	Scalar white = Scalar(255, 255, 255); //��� ����
	Scalar yellow(0, 255, 255);

	Mat image(400, 600, CV_8UC3, white);
	Point pt1(50, 130), pt2(200, 300), pt3(300, 150), pt4(400, 50); //��ǥ����
	Rect rect(pt3, Size(200, 150));

	line(image, pt1, pt2, red);//�����׸���
	line(image, pt3, pt4, green, 2, LINE_AA);//��Ƽ���ϸ��� ��
	line(image, pt3, pt4, green, 3, LINE_8, 1);//8���� ���ἱ, 1��Ʈ ����Ʈ

	rectangle(image, rect, blue, 2); //�簢�� �׸���
	rectangle(image, rect, blue, FILLED, LINE_4, 1); //4���� ���ἱ, 1��Ʈ ����Ʈ
	rectangle(image, pt1, pt2, red, 3);

	imshow("���� & �簢��", image);
	waitKey(0);

#endif

	//34 Page
#if 0
	Scalar orange(0, 165, 255), blue(255, 0, 0), magenta(255, 0, 255); //���󼱾�
	Mat image(300, 500, CV_8UC3, Scalar(255, 255, 255));

	Point center = (Point)image.size() / 2; //�����߽���ǥ
	Point pt1(70, 50), pt2(350, 220);

	circle(image, center, 100, blue);
	circle(image, pt1, 80, orange, 2);
	circle(image, pt2, 60, magenta, -1);

	int font = FONT_HERSHEY_COMPLEX;
	putText(image, "center_blue", center, font, 1.2, blue);
	putText(image, "pt1_orange", pt1, font, 0.8, orange);
	putText(image, "pt2_magenta", pt2 + Point(2, 2), font, 0.5, Scalar(0, 0, 0), 2);
	putText(image, "pt2_magenta", pt2, font, 0.5, magenta, 1);

	imshow("���׸���", image);
	waitKey(0);
#endif

	//39 Page
#if 0
	//
	//���������� �ʱ�ȭ�� 600 �� 400 ũ���� ���� ����
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
	Scalar blue(255, 0, 0); //���󼱾�
	Mat image = Mat(800, 600, CV_8UC3, Scalar(0, 0, 0));

	circle(image, Point(100, 100), 50, blue);
	imshow("Image", image);
	waitKey(0);

#endif

	//HW2
#if 0
	Mat image = Mat(400, 600, CV_8UC3, Scalar(255, 255, 255));
	int nDiameter = 200;
	Scalar blue(255, 0, 0), red(0, 0, 255); // ����
	Point center(image.cols / 2, image.rows / 2); // �±� �߽� ��ǥ
	Size circle_Big(nDiameter / 2, nDiameter / 2), circle_small(nDiameter / 4, nDiameter / 4); // �±� ������ �׸��� ���� �ΰ��� ��

	// ������ �Ķ����� �ݿ��� �Ʒ����� �׸���.
	ellipse(image, center, circle_Big, 0, 0, 180, blue, -1);
	ellipse(image, center, circle_Big, 0, 180, 360, red, -1);
	// ���� �ݿ��� �Ʒ� ���� �׷��ش�.
	ellipse(image, Point(center.x - circle_small.width, center.y), circle_small, 0, 0, 180, red, -1);
	ellipse(image, Point(center.x + circle_small.width, center.y), circle_small, 0, 180, 360, blue, -1);

	imshow("Image", image);
	waitKey(0);
#endif

	//3��
	//4 page
#if 0
	Mat image1(300, 400, CV_8U, Scalar(255));
	Mat image2(300, 400, CV_8U, Scalar(100));

	string title1 = "whiteâ ����";
	string title2 = "grayâ ����";

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
	namedWindow("Ű���� �̺�Ʈ", WINDOW_AUTOSIZE);
	imshow("Ű���� �̺�Ʈ", image);

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
			cout << "aŰ �Է�" << endl;
			break;
		case 'b':
			cout << "bŰ �Է�" << endl;
			break;
		case 0x41:
			cout << "AŰ �Է�" << endl;
			break;
		case 66:
			cout << "BŰ �Է�" << endl;
			break;
		case 0x250000:
			cout << "���� ȭ��ǥ Ű �Է�" << endl;
			break;
		case 0x260000:
			cout << "���� ȭ��ǥ Ű �Է�" << endl;
			break;
		case 0x270000:
			cout << "������ ȭ��ǥ Ű �Է�" << endl;
			break;
		case 0x280000:
			cout << "�Ʒ��� ȭ��ǥ Ű �Է�" << endl;
			break;
		default:
			cout << nkey << endl;
			break;
		}

	}
#endif

	//12 page
#if 0
	Mat img;
	img = imread("D:\\0.Job_Data\\1.��������\\3.����ΰ������а�\\23��2�б�(����2�б�)\\2.����ó������\\3.�ǽ�\\lenna.jpg", IMREAD_COLOR);
	if (img.empty())
	{
		cout << "������ ���� �� ����" << endl;
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
	Mat	src = imread("D:\\0.Job_Data\\1.��������\\3.����ΰ������а�\\23��2�б�(����2�б�)\\2.����ó������\\3.�ǽ�\\lenna.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "������ ���� �� ����" << endl;
	}
	imshow("src", src);
	while (1)
	{
		int	key = waitKeyEx();// ����ڷκ��� Ű�� ��ٸ���.
		cout << key << " ";

		if (key == 'q')// ����ڰ� ��q' �� ������ �����Ѵ�.
		{
			break;
		}
		else if (key == 2424832)
		{
			// ����ȭ��ǥ Ű
			src -= 50;// ������ ��ο�����
		}
		else if (key == 2555904)
		{
			// ������ȭ��ǥ Ű
			src += 50;// ������ �������.
		}
		imshow("src ", src);// ������ ����Ǿ����Ƿ� �ٽ� ǥ���Ѵ�
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
	Mat	src = imread("D:\\0.Job_Data\\1.��������\\3.����ΰ������а�\\23��2�б�(����2�б�)\\2.����ó������\\3.�ǽ�\\lenna.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "������ ���� �� ����" << endl;
	}
	imshow("src", src);

	setMouseCallback("src", onMouse, &src);
	waitKey(0);
#endif

	//26 page		
#if 0
	img_p26 = imread("D:\\0.Job_Data\\1.��������\\3.����ΰ������а�\\23��2�б�(����2�б�)\\2.����ó������\\3.�ǽ�\\lenna.jpg", IMREAD_COLOR);
	if (img_p26.empty())
	{
		cout << "������ ���� �� ����" << endl;
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
	createTrackbar("��Ⱚ", title, &nValue, 255, onChange);

	imshow(title, img_p30);
	waitKey(0);
#endif

	//33 page		
#if 0
	img_p33 = imread("D:\\0.Job_Data\\1.��������\\3.����ΰ������а�\\23��2�б�(����2�б�)\\2.����ó������\\3.�ǽ�\\lenna.jpg", IMREAD_COLOR);
	if (img_p33.empty())
	{
		cout << "������ ���� �� ����" << endl;
	}
	namedWindow("img", WINDOW_AUTOSIZE);
	imshow("img", img_p33);
	setMouseCallback("img", drawCircle);
	createTrackbar("R", "img", &red, 255, on_trackbar);
	createTrackbar("G", "img", &green, 255, on_trackbar);
	createTrackbar("B", "img", &blue, 255, on_trackbar);

	waitKey(0);
#endif

	//3�� HW1
#if 0
	img_hw1 = Mat(800, 600, CV_8UC1, Scalar(120));
	namedWindow("hw1", WINDOW_AUTOSIZE);
	imshow("hw1", img_hw1);
	setMouseCallback("hw1", hwDraw, &img_hw1);
	createTrackbar("Line", "hw1", &nLineThinkness, 10, on_trackbar_hw1);
	createTrackbar("Radius", "hw1", &nRadius, 200, on_trackbar_hw1);

	waitKey(0);
#endif
	//(3) - 27 Page
	//HW1
#if 0

	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "ī�޶� ���� ���� �ʾҽ��ϴ�." << endl;
		exit(1);
	}
	Rect roi(200, 100, 100, 200);
	Scalar red(0, 0, 255);

	for (;;)
	{
		Mat frame;
			
		capture.read(frame);
		
		Mat roiImage(frame, roi);
		
		roiImage += Scalar(0, 50, 0);

		rectangle(frame, roi, red, 3); //�簢�� �׸���
				
		imshow("ī�޶� ���� ����", frame);
		
		if (waitKey(30) >= 0) break;
	}

#endif
	//HW2
#if 0
	VideoCapture capture(0);
	if (!capture.isOpened());

	double fps = 15;
	int delay = cvRound(1000.0 / fps);
	Size size(640, 480);
	int fourcc = VideoWriter::fourcc('D', 'I', 'V', 'X');

	capture.set(CAP_PROP_FRAME_WIDTH, size.width);
	capture.set(CAP_PROP_FRAME_HEIGHT, size.height);

	VideoWriter writer;//���������� ���� ��ü

	//���� ���� �� ����
	writer.open("D:\\flip_test.avi", fourcc, fps, size);
	CV_Assert(writer.isOpened());

	for (;;)
	{
		Mat frame;
		capture >> frame; //ī�޶󿵻�ޱ�

		Mat xFlip;
		flip(frame, xFlip, 1);//�¿� flip
		writer << xFlip; //�������� ���������� ���� 

		imshow("ī�޶� ����", frame);
		imshow("ī�޶� Xflip", xFlip);
		if (waitKey(delay) >= 0)
		{
			break;
		}
	}
#endif

	//(4) - 35 Page
	//HW1
#if 0
	Range r1(2, 3), r2(3, 5);

	int data[] = {
		10,11,12,13,14,15,16,
		20,21,22,23,24,25,26,
		30,31,32,33,34,35,36,
		40,41,42,43,44,45,46,
	};

	Mat m1(5, 7, CV_32S, data);

	cout << m1 << endl;
	cout << m1(r1, r2) << endl;

	waitKey(0);
#endif
	//HW2
#if 0
	Mat array_100(10, 15, CV_16U, Scalar(100));	
	Rect roi_200(3, 1, 5, 4), roi_555(5, 3, 5, 4), roi_300(7, 5, 5, 4);

	Mat temp_1 = array_100(roi_200);
	temp_1 = Scalar(200);

	Mat temp_2 = array_100(roi_300);
	temp_2 = Scalar(300);

	Mat temp_3 = array_100(roi_555);
	temp_3 = Scalar(555);

	cout << array_100 << endl;
	

	waitKey(0);
#endif 
	
	//(5) - 20 Page
#if 0
	Mat image = imread("D:\\999.Image\\logo.jpg");

	Mat bgr[3], blue_img, red_img, green_img, zero(image.size(), CV_8U, Scalar(0));
	split(image, bgr);

	Mat buleImage[] = { bgr[0], zero, zero };
	Mat greenImage[] = { zero, bgr[1], zero };
	Mat redImage[] = { zero, zero, bgr[2] };

	merge(buleImage, 3, blue_img);
	merge(greenImage, 3, green_img);
	merge(redImage, 3, red_img);

	imshow("image", image);
	imshow("blue_img", blue_img);
	imshow("red_img", red_img);
	imshow("green_img", green_img);
	waitKey(0);
#endif

	//(5) - 21 Page
#if 0	
	namedWindow("test", WINDOW_NORMAL);
	resizeWindow("test", 400, 300);

	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "ī�޶� ���� ���� �ʾҽ��ϴ�." << endl;
		exit(1);
	}		

	Rect roi(30, 30, 320, 240);
	Scalar red(0, 0, 255);

	Mat backImage(300, 400, CV_8UC3, Scalar(0,0,0));

	capture.set(CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CAP_PROP_FRAME_HEIGHT, 240);

	while(true)
	{
		Mat frame;

		capture.read(frame);

		Mat roiImage(backImage, roi);

		frame.copyTo(roiImage);

		rectangle(backImage, roi, red, 3); //�簢�� �׸���

		imshow("test", backImage);
		
		if (waitKey(30) >= 0) break;
	}
#endif

	//ȭ��ó�� - p.38
	//HW1
#if 0 
	img_Pixel_Processing = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	if (img_Pixel_Processing.empty())
	{
		cout << "������ ���� �� ����" << endl;
	}
	imshow("image", img_Pixel_Processing);
	setMouseCallback("image", onMouse_Pixel_Processing, 0);
	waitKey(0);
#endif
	//HW2
#if 0
	
	img_Pixel_Processing_src1 = imread("D:\\999.Image\\lenna.jpg");
	img_Pixel_Processing_src2 = imread("D:\\999.Image\\bug.jpg");

	Size sz1(300, 300);
	resize(img_Pixel_Processing_src1, img_Pixel_Processing_src1, sz1);
	resize(img_Pixel_Processing_src2, img_Pixel_Processing_src2, sz1);
	
	namedWindow(title_Pixel_Processing_HW2, WINDOW_AUTOSIZE);
	createTrackbar("Alpha", title_Pixel_Processing_HW2, &nAlpha_Pixel_Processing, 10, on_trackbar_Pixel_Processing);

	waitKey();
#endif
	//������׷� - p.31
	//HW2
#if 0
	Mat src_Image = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	Mat temp_row;
	Mat temp_col;

	int nDim = 0; //0 - row(��), 1 - col(->)
	reduce(src_Image, temp_row, 0, REDUCE_SUM, CV_32F);

	reduce(src_Image, temp_col, 1, REDUCE_SUM, CV_32F);

	int hist_row[400] = { 0 };
	int hist_col[400] = { 0 };

	for (int y = 0; y < temp_row.rows; y++)
	{
		for (int x = 0; x < temp_row.cols; x++)
		{
			hist_row[x] = (int)temp_row.at<float>(y, x);
		}
	}

	for (int y = 0; y < temp_col.rows; y++)
	{
		for (int x = 0; x < temp_col.cols; x++)
		{
			hist_col[y] = (int)temp_col.at<float>(y, x);
		}
	}
	int hist_w = src_Image.cols; //������׷� ������ ��
	int hist_h = src_Image.rows; //������׷� ������ ����
	int bin_w = cvRound((double)hist_w / 256); //���� ��

	//������׷��� �׷����� ����(Į��� ����)
	Mat histImage_row(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat histImage_col(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	//������׷��� �ִ밪�� ã�´�.
	int max_row = hist_row[0];
	for (int i = 1; i < 400; i++)
	{
		if (max_row < hist_row[i])
		{
			max_row = hist_row[i];
		}
	}
	int max_col = hist_col[0];
	for (int i = 1; i < 400; i++)
	{
		if (max_col < hist_col[i])
		{
			max_col = hist_col[i];
		}
	}

	//������׷� �迭�� �ִ밪���� ����ȭ �Ѵ�.(�ִ밪�� �ִ���̰� �ǵ���)
	for (int i = 0; i < 399; i++)
	{
		hist_row[i] = floor(((double)hist_row[i] / max_row) * histImage_row.rows);
	}

	for (int i = 0; i < 399; i++)
	{
		hist_col[i] = floor(((double)hist_col[i] / max_col) * histImage_col.rows);
	}

	//������׷��� ���� ������ ����� �׸���.
	for (int i = 0; i < 399; i++)
	{
		line(histImage_row, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - hist_row[i]), Scalar(100, 20, 100));
	}

	for (int i = 0; i < 399; i++)
	{
		line(histImage_col, Point(bin_w * (i), hist_h), Point(bin_w * (i), hist_h - hist_col[i]), Scalar(100, 100, 100));
	}

	imshow("src_Image", src_Image);

	imshow("Histogram_row", histImage_row);

	rotate(histImage_col, histImage_col, ROTATE_90_COUNTERCLOCKWISE);
	imshow("Histogram_col", histImage_col);
	
	waitKey();
#endif
	//HW3
#if 0
	img_Histogram  = imread("D:\\999.Image\\lenna.jpg");
	if (img_Histogram.empty())
	{
		return -1;
	}
	imshow("image", img_Histogram);

	setMouseCallback("image", onMouse_Histogram, 0);
	waitKey();
#endif

	//10-�������͸� - HW1
#if 0
	Mat src, src_gray, dst, dst_Nofilter;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	src = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
	{
		return -1;
	}
	Mat src_Nofilter = src.clone();
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat abs_dst;
	Mat abs_dst_Nofilter;
	Laplacian(src, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	Laplacian(src_Nofilter, dst_Nofilter, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, abs_dst);
	convertScaleAbs(dst_Nofilter, abs_dst_Nofilter);
	imshow("Original", src_Nofilter);
	imshow("GaussianBlur", src);
	imshow("Laplacian", abs_dst);
	imshow("Nofilter", abs_dst_Nofilter);
	waitKey();
#endif

	//10-�������͸� - HW2
#if 0
	Mat src = imread("D:\\999.Image\\city1.jpg", IMREAD_GRAYSCALE);
	if (src.empty())
	{
		return -1;
	}
	Mat dst(src.size(), src.type());
	Mat noise_img = Mat::zeros(src.rows, src.cols, CV_8U);
	randu(noise_img, 0, 255);//noise_img�� ���ȭ�Ҹ� 0~255������ ������ ä��

	Mat black_img = noise_img < 10; // noise_img�� ȭ�Ұ��� 10 ���� ������ 1�̵Ǵ� black_img ����
	Mat white_img = noise_img > 245; // noise_img�� ȭ�Ұ��� 245 ���� ũ�� 1�̵Ǵ� white_img ����

	Mat src1 = src.clone();
	src1.setTo(255, white_img); //white_img�� ȭ�Ұ��� 1�̸� src1�� ȭ�Ұ��� 255�� �Ѵ�. salt noise
	src1.setTo(0, black_img); //black_img�� ȭ�Ұ��� 1�̸� src1�� ȭ�Ұ��� 0�� �Ѵ�. pepper noise
	medianFilter(src1, dst, 5);
	imshow("srorce", src1);
	imshow("result", dst);
	waitKey();
#endif

	//11- ����������ȯ - HW1
#if 0
	_11_HW1_src = imread("D:\\999.Image\\lenna.jpg");
	warp_dst = Mat::zeros(_11_HW1_src.rows, _11_HW1_src.cols, _11_HW1_src.type());
	imshow("11-HW1-SRC", _11_HW1_src);
	imshow("11-HW1-DST", warp_dst);
	setMouseCallback("11-HW1-SRC", onMouse_11_HW1_src, 0);
	setMouseCallback("11-HW1-DST", onMouse_11_HW1_dst, 0);
	waitKey();
#endif

	//11- ����������ȯ - HW2
#if 0
	_11_HW2_src = imread("D:\\999.Image\\book.jpg");
	perspective_dst = Mat::zeros(_11_HW2_src.rows, _11_HW2_src.cols, _11_HW2_src.type());
	imshow("11-HW2-SRC", _11_HW2_src);
	imshow("11-HW2-DST", perspective_dst);
	setMouseCallback("11-HW2-SRC", onMouse_11_HW2_src, 0);
	setMouseCallback("11-HW2-DST", onMouse_11_HW2_dst, 0);
	waitKey();
#endif

	//8�� - �������� ó�� - HW 1
#if 0
	uchar data[] = { 0,0,0,0,0,0,0,0,
					0,0,0,1,1,0,0,0,
					0,1,1,1,1,1,1,0,
					0,0,1,1,1,0,1,0,
					0,1,1,1,1,1,1,0,
					0,0,0,0,0,0,0,0,
	};
	uchar maskdata[] = { 0,1,0,
						1,1,1,
						0,1,0, };
	Mat srcImg(6, 8, CV_8UC1, data);
	Mat mask(3, 3, CV_8UC1, maskdata);

	cout << srcImg << endl;
	cout << mask << endl;

	Mat dst1;
	morphologyEx(srcImg, dst1, MORPH_DILATE, mask);

	cout << "HW1" << endl;
	cout << dst1 << endl;
	waitKey();
#endif
	//8�� - �������� ó�� - HW 2
#if 0
	uchar data[] = { 0,0,0,0,0,0,0,0,
				0,0,0,1,1,0,0,0,
				0,1,1,1,1,1,1,0,
				0,0,1,1,1,0,1,0,
				0,1,1,1,1,1,1,0,
				0,0,0,0,0,0,0,0,
	};
	uchar maskdata[] = { 0,1,0,
						1,1,1,
						0,1,0, };
	Mat srcImg(6, 8, CV_8UC1, data);
	Mat mask(3, 3, CV_8UC1, maskdata);

	cout << srcImg << endl;
	cout << mask << endl;

	Mat dst1;
	morphologyEx(srcImg, dst1, MORPH_ERODE, mask);

	cout << "HW2" << endl;
	cout << dst1 << endl;
	waitKey();
#endif

	//8�� - �������� ó�� - HW 3
#if 0
	uchar data[] = { 0,0,0,0,0,0,
					 0,0,1,0,0,0,
					 0,1,1,0,0,0,
					 0,1,1,1,1,0,
					 0,1,1,0,1,0,
					 0,0,0,0,0,0,
	};
	uchar maskdata[] = { 0,1,
						1,1,
	};
	Mat srcImg(6, 6, CV_8UC1, data);
	Mat mask(2, 2, CV_8UC1, maskdata);

	cout << srcImg << endl;
	cout << mask << endl;

	Mat dst1;
	morphologyEx(srcImg, dst1, MORPH_OPEN, mask);

	cout << "HW3" << endl;
	cout << dst1 << endl;
#endif

	//8�� - �������� ó�� - HW 4
#if 0
	uchar data[] =
	{   0,0,0,0,0,1,0,0,
		0,1,1,1,0,0,0,0,
		0,1,1,1,0,0,1,1,
		0,1,1,1,0,1,1,1,
		1,1,1,0,0,1,0,1,
		0,0,0,0,1,1,1,1,
		0,0,0,0,1,1,1,0,
		0,0,0,0,0,0,0,0,
	};
	uchar maskdata[] = { 0,1,0,
						1,1,1,
						0,1,0, };
	Mat srcImg(6, 8, CV_8UC1, data);
	Mat mask(3, 3, CV_8UC1, maskdata);

	cout << srcImg << endl;
	cout << mask << endl;

	Mat dst_Open, dst_Close;
	morphologyEx(srcImg, dst_Open, MORPH_OPEN, mask);
	morphologyEx(srcImg, dst_Close, MORPH_CLOSE, mask);

	cout << "HW4" << endl;
	cout << "Open" << endl;
	cout << dst_Open << endl;
	cout << "Close" << endl;
	cout << dst_Close << endl;
	waitKey();
#endif

	//8�� - �������� ó�� - HW 5
#if 0
	Mat image = imread("D:\\999.Image\\coins.jpg", IMREAD_COLOR);

	Mat gray, sobel, th_img, morph;
	Mat kernel(5, 5, CV_8UC1, Scalar(1));		// ���� ���� ����ũ
	cvtColor(image, gray, COLOR_BGR2GRAY);		// ��ϵ� ���� ��ȯ

	blur(gray, gray, Size(5, 5));				// ����

	threshold(gray, th_img, 90, 255, THRESH_BINARY);	// ����ȭ ����
	morphologyEx(th_img, morph, MORPH_CLOSE, kernel);	// ���� ���� ����

	imshow("image", image);
	imshow("morph", morph);
	waitKey();
#endif

	//9�� - Į�󿵻�ó�� - HW1
#if 0
	img_9_HW = imread("D:\\999.Image\\color_space.jpg");
	imshow("image_9_HW", img_9_HW);
	Mat clone = img_9_HW.clone();
	setMouseCallback("image_9_HW", onMouse_9_Color_Processing);
	waitKey();
#endif 
	//9�� - Į�󿵻�ó�� - HW2
#if 0
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		return -1;
	}
	for (;;)
	{
		Mat img_HSV;
		Mat frame;
		capture >> frame;
		cvtColor(frame, img_HSV, COLOR_BGR2HSV);

		Mat imgThreshold;
		Scalar lowerLimit = Scalar(5, 20, 20);
		Scalar upperLimit = Scalar(20, 150, 150);

		inRange(img_HSV, lowerLimit, upperLimit, imgThreshold);
		
		Mat dst;
		bitwise_and(frame, frame, dst, imgThreshold = imgThreshold);
		imwrite("d:\\test.bmp", frame);
		imshow("frame", frame);
		//imshow("imgThreshold", imgThreshold);
		imshow("HSV", dst);
		if (waitKey(30) >= 0)
		{
			break;
		}
	}
	waitKey();
#endif
	//14 - ���ļ����� ó�� - HW1
#if 0
	Mat src = imread("D:\\999.Image\\image.jpg", IMREAD_GRAYSCALE);
	Mat src_float;

	// �׷��̽����� ������ �Ǽ� �������� ��ȯ�Ѵ�.
	src.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	Mat dft_image;
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);
	shuffleDFT(dft_image);
	displayDFT(dft_image);
	imshow("src", src);
	waitKey();
#endif

	//14 - ���ļ����� ó�� - HW2
#if 0
	Mat img = imread("D:\\999.Image\\lenna.jpg", IMREAD_GRAYSCALE);
	imshow("Img_Ori", img);
	Mat img_Clone;
	img_Clone = img.clone();

	Mat src_float;
	img.convertTo(src_float, CV_32FC1, 1.0 / 255.0);
	Mat dft_image;
	dft(src_float, dft_image, DFT_COMPLEX_OUTPUT);
	shuffleDFT(dft_image);
	displayDFT(dft_image, "dft");

	Mat lowpass = getFilter_Circle(dft_image.size());
	displayDFT(lowpass, "lowpass");
	Mat result;

	multiply(dft_image, lowpass, result);

	Mat inverted_image;
	shuffleDFT(result);
	idft(result, inverted_image, DFT_SCALE | DFT_REAL_OUTPUT);
	imshow("Frequency_lowpass", inverted_image);

	
	Mat dst;
	blur(img_Clone, dst, Size(9, 9));
	imshow("dst", dst);

	waitKey();
#endif

	//15 - ������� - HW1
#if 0
	Mat img = imread("D:\\999.Image\\keyboard.bmp", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}
	imshow("img_ori", img);
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);

	GaussianBlur(gray, gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

	Mat bin;
	threshold(gray, bin, 200, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < (int)contours.size(); i++)
	{
		Rect  mr = boundingRect(contours[i]);
		
		rectangle(img, mr, Scalar(0, 255, 255), 1);
	}
	
	imshow("img", img);
	waitKey();
#endif

	//15 - ������ - HW2
#if 0
	Mat img = imread("D:\\999.Image\\shape.bmp", IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "Image load failed!" << endl;
		return -1;
	}
	Mat img_Clone;
	img_Clone = img.clone();

	Mat gray;
	cvtColor(img_Clone, gray, COLOR_BGR2GRAY);

	Mat bin;
	threshold(gray, bin, 250, 255, THRESH_BINARY_INV);

	vector<vector<Point>> contours;
	findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	drawContours(img_Clone, contours, -1, Scalar(0, 0, 0), 3);
	imshow("img", img);
	imshow("img_Clone", img_Clone);
	waitKey();
#endif

	//16 - Ư¡����(1) - HW3
#if 0
	Mat src = imread("D:\\999.Image\\hough_test3.jpg", IMREAD_COLOR);
	if (src.empty())
	{
		cout << "can not open " << endl;
		return -1;
	}

	Mat dst, cdst;
	Canny(src, dst, 100, 200);
	cvtColor(dst, cdst, COLOR_GRAY2BGR);

	vector<Vec4i> lines;
	HoughLinesP(dst, lines, 1, (CV_PI / 180) * 2, 50, 100, 20);
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 2, LINE_AA);
	}

	imshow("source", src);
	imshow("detected lines", cdst);
	waitKey();
#endif

	//17 - Ư¡����(2) - HW1
#if 0
	Mat img = imread("D:\\999.Image\\building.jpg", IMREAD_COLOR);
	if (img.empty())
	{
		cout << "can not open " << endl;
		return -1;
	}
	
	//Harris
	Mat Harris_gray;
	Mat Harris_img;
	Mat Harris_Result_img;
	int blockSize = 4;
	int apertureSize = 3;
	double k = 0.04;
	int  thresh = 20;

	imshow("img", img);

	cvtColor(img, Harris_gray, COLOR_BGR2GRAY);
	
	cornerHarris(Harris_gray, Harris_img, blockSize, apertureSize, k);	// OpenCV ���� �Լ�
	Harris_Result_img = draw_coner(Harris_img, img.clone(), thresh);
	imshow("Harris", Harris_Result_img);

	//FAST
	vector<KeyPoint> keypoints_FAST;
	Mat FAST_gray;
	Mat FAST_Result_img;
	cvtColor(img, FAST_gray, COLOR_BGR2GRAY);
	FAST(FAST_gray, keypoints_FAST, 60, true);
	
	cvtColor(FAST_gray, FAST_Result_img, COLOR_GRAY2BGR);

	for (KeyPoint kp : keypoints_FAST)
	{
		Point pt(cvRound(kp.pt.x), cvRound(kp.pt.y));
		circle(FAST_Result_img, pt, 5, Scalar(0, 0, 255), 2);
	}
	imshow("FAST", FAST_Result_img);

	//SIFT
	Mat SIFT_gray;
	Mat SIFT_desc;
	Mat SIFT_Result_img;
	Ptr<Feature2D> feature_SIFT = SIFT::create();
	vector<KeyPoint> keypoints_SIFT;

	cvtColor(img, SIFT_gray, COLOR_BGR2GRAY);
	feature_SIFT->detect(SIFT_gray, keypoints_SIFT);
	feature_SIFT->compute(SIFT_gray, keypoints_SIFT, SIFT_desc);
	drawKeypoints(SIFT_gray, keypoints_SIFT, SIFT_Result_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SIFT", SIFT_Result_img);

	//SURF
	Mat SURF_gray;
	Mat SURF_desc;
	Mat SURF_Result_img;
	Ptr<Feature2D> feature_SURF = xfeatures2d::SURF::create();
	vector<KeyPoint> keypoints_SURF;
	cvtColor(img, SURF_gray, COLOR_BGR2GRAY);
	feature_SURF->detect(SURF_gray, keypoints_SURF);
	feature_SURF->compute(SURF_gray, keypoints_SURF, SURF_desc);
	drawKeypoints(SURF_gray, keypoints_SURF, SURF_Result_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("SURF", SURF_Result_img);
	
	//ORB 
	Mat ORB_gray;
	Mat ORB_desc;
	Mat ORB_Result_img;
	Ptr<Feature2D> feature_ORB = ORB::create();
	vector<KeyPoint> keypoints_ORB;
	cvtColor(img, ORB_gray, COLOR_BGR2GRAY);
		
	feature_ORB->detect(ORB_gray, keypoints_ORB);
	feature_ORB->compute(ORB_gray, keypoints_ORB, ORB_desc);

	drawKeypoints(ORB_gray, keypoints_ORB, ORB_Result_img, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("ORB", ORB_Result_img);

	waitKey();
#endif

	//18-Ư¡��Ī - HW1
#if 0
	template_matching();
#endif

	//18-Ư¡��Ī - HW2
#if 0
	Keypoint_Matching();
#endif
	//19-����з� - HW3
#if 0
	Mat src = imread("D:\\999.Image\\lenna.jpg", 1);

	Mat hsv_img;
	cvtColor(src, hsv_img, COLOR_BGR2HSV);

	imshow("src image", src);
	imshow("hsv_img image", hsv_img);
	// �н� �����͸� �����. 
	Mat samples(hsv_img.rows * hsv_img.cols, 3, CV_32F);
	for (int y = 0; y < hsv_img.rows; y++)
	{
		for (int x = 0; x < hsv_img.cols; x++)
		{
			for (int z = 0; z < 3; z++)
			{
				samples.at<float>(y + x * hsv_img.rows, z) = hsv_img.at<Vec3b>(y, x)[z];
			}
		}
	}

	// Ŭ�������� ������ 15�� �ȴ�. 
	int clusterCount = 10;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat new_image(hsv_img.size(), hsv_img.type());
	for (int y = 0; y < hsv_img.rows; y++)
	{
		for (int x = 0; x < hsv_img.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * hsv_img.rows, 0);
			new_image.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
			new_image.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
			new_image.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);
		}
	}
	imshow("clustered image", new_image);
	waitKey(0);
#endif

	return 0;
}