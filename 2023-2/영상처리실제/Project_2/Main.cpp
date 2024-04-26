#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	Mat img[20];
	string strSavefilePath;
	
	Ptr<Feature2D> orb = ORB::create();
	Ptr<DescriptorMatcher> matcher = BFMatcher::create(NORM_HAMMING);
	vector<KeyPoint> keypoints1[20];
	vector<KeyPoint> keypoints2[20];
	Mat desc1[20];
	Mat desc2[20];
	vector<DMatch> matches[20];
	Mat dst;
	double test = 0;
	double dFirstAngle = 0;
	double dFirstpt_x = 0;
	double dFirstpt_y = 0;
	double dResult_X = 0.0;
	double dResult_Y = 0.0;
	double dResuly_Yaw = 0.0;
	double dx_Temp = 0.0;
	double dy_Temp = 0.0;
	double angle_temp = 0.0;

	img[0]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000000.png", IMREAD_GRAYSCALE);
	img[1]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000005.png", IMREAD_GRAYSCALE);
	img[2]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000010.png", IMREAD_GRAYSCALE);
	img[3]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000015.png", IMREAD_GRAYSCALE);
	img[4]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000020.png", IMREAD_GRAYSCALE);
	img[5]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000025.png", IMREAD_GRAYSCALE);
	img[6]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000030.png", IMREAD_GRAYSCALE);
	img[7]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000035.png", IMREAD_GRAYSCALE);
	img[8]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000040.png", IMREAD_GRAYSCALE);
	img[9]   = imread("D:\\999.Image\\Car_Pos_Image\\05\\000045.png", IMREAD_GRAYSCALE);
	img[10]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000050.png", IMREAD_GRAYSCALE);
	img[11]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000055.png", IMREAD_GRAYSCALE);
	img[12]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000060.png", IMREAD_GRAYSCALE);
	img[13]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000065.png", IMREAD_GRAYSCALE);
	img[14]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000070.png", IMREAD_GRAYSCALE);
	img[15]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000075.png", IMREAD_GRAYSCALE);
	img[16]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000080.png", IMREAD_GRAYSCALE);
	img[17]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000085.png", IMREAD_GRAYSCALE);
	img[18]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000090.png", IMREAD_GRAYSCALE);
	img[19]  = imread("D:\\999.Image\\Car_Pos_Image\\05\\000095.png", IMREAD_GRAYSCALE);	
	strSavefilePath = "D:\\999.Image\\Car_Pos_Image\\05\\odometry.txt";	
	ofstream writeFile(strSavefilePath.data());

	for (int i = 0; i < 20; i++)
	{
		if (i == 0)
		{
			orb->detectAndCompute(img[i], Mat(), keypoints1[i], desc1[i]);
			orb->detectAndCompute(img[i], Mat(), keypoints2[i], desc2[i]);
		}
		else
		{			
			orb->detectAndCompute(img[i - 1], Mat(), keypoints1[i], desc1[i]);
			orb->detectAndCompute(img[i], Mat(), keypoints2[i], desc2[i]);
		}

		//matcher->match(질의기술자, 훈련기술자, 매칭결과);		
		matcher->match(desc1[i], desc2[i], matches[i]);

		std::sort(matches[i].begin(), matches[i].end()); // Distance 오름차순 정렬

		vector<DMatch> good_matches(matches[i].begin(), matches[i].begin() + 50); // 상위 50개만 선별
#if 0
		drawMatches(img[i], keypoints1[i], img[i + 1], keypoints2[i], good_matches, dst,
			Scalar::all(-1), Scalar::all(-1), vector<char>(),
			DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


		for (size_t j = 0; j < good_matches.size(); j++)
		{
			pts1[i].push_back(keypoints1[i][good_matches[j].queryIdx].pt);
			pts2[i].push_back(keypoints2[i][good_matches[j].trainIdx].pt);

			angle1[i].push_back(keypoints1[i][good_matches[j].queryIdx].angle);
			angle2[i].push_back(keypoints2[i][good_matches[j].trainIdx].angle);
		}
#endif

		dx_Temp = (double)keypoints1[i][good_matches[0].queryIdx].pt.x - 
				  (double)keypoints2[i][good_matches[0].trainIdx].pt.x;
		dy_Temp = (double)keypoints1[i][good_matches[0].queryIdx].pt.y - 
			      (double)keypoints2[i][good_matches[0].trainIdx].pt.y;
		angle_temp = ((double)keypoints1[i][good_matches[0].queryIdx].angle -
			          (double)keypoints2[i][good_matches[0].trainIdx].angle) * CV_PI / 180;
			
		if (i == 0)
		{
			dFirstpt_x = dx_Temp;
			dFirstpt_y = dy_Temp;

			dFirstAngle = angle_temp;
		}

		dx_Temp = (dFirstpt_x + dx_Temp) / 100; //초기값에서 다음 값 계산
		dx_Temp = (round(dx_Temp * 100) / 100); // 소수 2번째짜리 까지 처리
		dy_Temp = (dFirstpt_y + dy_Temp) / 100; //초기값에서 다음 값 계산
		dy_Temp = (round(dy_Temp * 100) / 100); // 소수 2번째짜리 까지 처리

		angle_temp = dFirstAngle + angle_temp;
		angle_temp = (round(angle_temp * 100) / 100); // 소수 2번째짜리 까지 처리

		dResult_X = dResult_X + dx_Temp;  //최종 결과값에 누적
		dResult_Y = dResult_Y + dy_Temp;
		dResuly_Yaw = dResuly_Yaw + angle_temp;

		if (writeFile.is_open())
		{
			writeFile << fixed;
			writeFile.precision(2);
			
			writeFile << dResult_X << "\t" << dResult_Y << "\t" << dResuly_Yaw << "\n";
		}		

		//string steTemp = "dst" + to_string(i);

		//imshow(steTemp, dst);
	}

	writeFile.close();

	waitKey(0);

	return 0;
}