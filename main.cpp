#include<iostream>   
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2\opencv.hpp>
#include "elas.h"

#include <time.h>

#pragma comment(lib, "opencv_world341.lib")


using namespace std;
using namespace cv;

int ElasMatch(cv::Mat leftImage, cv::Mat rightImage);

int main() {

	Mat left_org = imread("./img/mL2.jpg",IMREAD_GRAYSCALE);
	Mat right_org = imread("./img/mR2.jpg", IMREAD_GRAYSCALE);
	Mat left, right;
	resize(left_org, left, Size(left_org.cols/2., left_org.rows/2.));
	resize(right_org, right, Size(right_org.cols/2., right_org.rows/2.));


	//Mat left_org = imread("./img/L2.jpg", IMREAD_GRAYSCALE);
	//Mat right_org = imread("./img/R2.jpg", IMREAD_GRAYSCALE);
	//Mat left, right;
	//resize(left_org, left, Size(left_org.cols / 5., left_org.rows / 5.));
	//resize(right_org, right, Size(right_org.cols / 5., right_org.rows / 5.));

	ElasMatch(left, right);
	//Mat left_disp = imread("1L_disp.pgm");
	//imshow("left",left_disp);

	cvWaitKey(0);
	return 0;
}

int ElasMatch(cv::Mat leftImage, cv::Mat rightImage)
{
	cv::Mat disp_l, disp_r, disp8u_l, disp8u_r;
	double minVal; double maxVal; 

	// generate disparity image using LIBELAS
	int bd = 0;
	const int32_t dims[3] = { leftImage.cols,leftImage.rows,leftImage.cols };
	cv::Mat leftdpf = cv::Mat::zeros(cv::Size(leftImage.cols, leftImage.rows), CV_32F);
	cv::Mat rightdpf = cv::Mat::zeros(cv::Size(leftImage.cols, leftImage.rows), CV_32F);
	Elas::parameters param;
	//param.postprocess_only_left = false;
	Elas elas(param);

	clock_t start = clock();
	elas.process(leftImage.data, rightImage.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
	cout << clock() - start << endl;

	cv::Mat(leftdpf(cv::Rect(bd, 0, leftImage.cols, leftImage.rows))).copyTo(disp_l);
	cv::Mat(rightdpf(cv::Rect(bd, 0, rightImage.cols, rightImage.rows))).copyTo(disp_r);

	//-- Check its extreme values
	cv::minMaxLoc(disp_l, &minVal, &maxVal);
	cout << "Min disp: Max value" << minVal << maxVal; //numberOfDisparities.= (maxVal - minVal)

	//-- Display it as a CV_8UC1 image
	disp_l.convertTo(disp8u_l, CV_8U, 255 / (maxVal - minVal));//(numberOfDisparities*16.)

	cv::minMaxLoc(disp_r, &minVal, &maxVal);
	cout << "Min disp: Max value" << minVal << maxVal; //numberOfDisparities.= (maxVal - minVal)

	//-- Display it as a CV_8UC1 image
	disp_r.convertTo(disp8u_r, CV_8U, 255 / (maxVal - minVal));//(numberOfDisparities*16.)

	cv::normalize(disp8u_l, disp8u_l, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image
	cv::normalize(disp8u_r, disp8u_r, 0, 255, CV_MINMAX, CV_8UC1);    // obtain normalized image

	cv::imshow("Left", leftImage);
	cv::imshow("Right", rightImage);

	cv::imshow("Elas_left", disp8u_l);
	cv::imshow("Elas_right", disp8u_r);
	cv::imwrite("Elas_left.png", disp8u_l);
	cv::imwrite("Elas_right.png", disp8u_r);

	cout << endl << "Over" << endl;
	cv::waitKey(0);

	return 0;
}
