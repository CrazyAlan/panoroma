#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

/// Global variables
Mat src_1, src_gray_1;
Mat src_2, src_gray_2;
int thresh = 200;
int max_thresh = 255;

const char* source_window = "Source image";
const char* corners_window = "Sift detected";
const char* matches_window = "Good Matches";

const float rows = 1200;
const float cols = 1600;

const float pano_rows = 2400;
const float pano_cols = 4800;

const float offset_rows = 600;
const float offset_cols = 1200;

float norm_ma[3][3] = {{2/cols,0,-1},{0,2/rows,-1},{0,0,1}};
Mat norm_matrix = Mat(3, 3, CV_32FC1,norm_ma);

/// Function header
void siftDetector( int, void* );
Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches,Mat* X_norm1, Mat* X_norm2);
Mat buildCoord(void);

/**
 * @function main
 */
int main( int, char** argv )
{
    /// Load source image and convert it to grayn
    src_1 = imread( argv[1], 1 );
    cvtColor( src_1, src_gray_1, COLOR_BGR2GRAY );
    src_2 = imread(argv[2], 1);
    cvtColor(src_2, src_gray_2, COLOR_BGR2GRAY);
    
    /// Create a window and a trackbar
    namedWindow( source_window, CV_WINDOW_NORMAL );
    resizeWindow( source_window, 0.1 ,0.1);
    imshow( source_window, src_gray_1 );
    
    siftDetector( 0, 0 );
    
    waitKey(0);
    return(0);
}

/**
 * @function siftDetector
 * @brief Executes the sift detection and draw a circle around the possible keypoints
 */
void siftDetector( int, void* )
{
    SIFT detector = SIFT(thresh,3,0.04,10,1.6);
    vector<cv::KeyPoint> keypoints_1;
    vector<cv::KeyPoint> keypoints_2;
    Mat descriptor_1;
    Mat descriptor_2;
    detector.operator()(src_gray_1, Mat(), keypoints_1,descriptor_1,false);
    detector.operator()(src_gray_2,Mat(),keypoints_2,descriptor_2,false);
    
    //Feature Matching
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor_1, descriptor_2, matches);
    double max_dist = 0; double min_dist = 100;
    
    //Caculate max and min distances between keypoints
    for (int i =0 ; i < descriptor_1.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    //Draw Good Matches
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < descriptor_1.rows; i++ )
    { if( matches[i].distance <= max(3*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
    }
    
    Mat img_matches;
    drawMatches(src_1, keypoints_1, src_2, keypoints_2, good_matches, img_matches);
    
    Mat X_norm1;
    Mat X_norm2;
    Mat A = buildA(keypoints_1, keypoints_2, good_matches,&X_norm1,&X_norm2);
    SVD svd(A);
    Mat h = svd.vt.t().col(svd.vt.rows - 1);
    Mat H = h.t();
    H = H.reshape(1, 3);
    cout << "h is "  << h << endl;
    cout << "H is " << H << endl;
    
    Mat coord = buildCoord();
    Mat coord_norm = norm_matrix*coord;
    Mat h_x = H*coord_norm;
    Mat H_X = norm_matrix.inv()*h_x;
    
    for (int i =0; i<(int)cols*(int)rows; i++) {
        H_X.col(i) /= H_X.at<float>(2, i);
    }
    
    H_X.row(0) += (float)offset_cols;
    H_X.row(1) += (float)offset_rows;
    
    Mat out_img1((int)pano_rows,(int)pano_cols,CV_8UC1,Scalar::all(0));
    Mat out_img2((int)pano_rows,(int)pano_cols,CV_8UC1,Scalar::all(0));
    src_gray_2.copyTo(out_img2(Range((int)offset_rows,(int)rows+(int)offset_rows),Range((int)offset_cols,(int)offset_cols+(int)cols)));
    
    
    
    //cout << "Src Gray"<< endl<< src_gray_2.at<float>(1197, 1452)<< endl;
    for (int i=0; i<(int)rows; i++) {
        for (int j=0; j<(int)cols; j++) {
            int x = H_X.at<float>(0,i*(int)cols + j);
            int y = H_X.at<float>(1,i*(int)cols + j);
           // cout << src_gray_1.at<uchar>(i, j) << endl;
            out_img1.at<uchar>(y,x) = src_gray_1.at<uchar>(i, j);
          
            
        }
    }
    
    double alpha = 0.5;
    double beta = 1 - alpha;
    Mat dst;
    addWeighted(out_img1, alpha, out_img2, beta, 1, dst);
    
    
    cout << "Coord 1 " << endl << H_X(Range(0,3),Range(0, 10)) << endl;
    resize(dst , dst, Size(dst.cols/4,out_img1.rows/4));
    namedWindow(matches_window,0);
    imshow( "Good Matches", dst);
    
    
}

/** 
 * @function buildA
 * @build Matrix A for DLT algorithm
 */

Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches,Mat* X_norm1,Mat* X_norm2)
{
    // Creat Coordinate Matrix
    Mat X_img1(3,(int)good_matches.size(),CV_32FC1);
    Mat X_img2(3,(int)good_matches.size(),CV_32FC1);
    for (int i=0; i<(int)good_matches.size(); i++) {
        int id1 = good_matches[i].queryIdx;
        int id2 = good_matches[i].trainIdx;
        float x[3] = {keypoint_1[id1].pt.x,keypoint_1[id1].pt.y,1};
        float x_prime[3] = {keypoint_2[id2].pt.x,keypoint_2[id2].pt.y,1};
        Mat(3,1,CV_32FC1,x).copyTo(X_img1.col(i));
        Mat(3, 1, CV_32FC1, x_prime).copyTo(X_img2.col(i));
    }
    
    *X_norm1 = norm_matrix*X_img1;
    *X_norm2 = norm_matrix*X_img2;
    
    // Creat 8*9 Matrix, using 4 points
    Mat A(2*(int)good_matches.size(),9,CV_32FC1,Scalar::all(0));
    for (int i=0; i<good_matches.size(); i++) {
        //cout <<(*X_norm1).col(i).t() << endl;
        A(Range(i*2,i*2+1),Range(3,6)) = -(*X_norm2).at<float>(2,i) * (*X_norm1).col(i).t();
        A(Range(i*2,i*2+1),Range(6,9)) = (*X_norm2).at<float>(1,i) * (*X_norm1).col(i).t();
        A(Range(i*2+1,i*2+2),Range(0,3)) = (*X_norm2).at<float>(2,i) * (*X_norm1).col(i).t();
        A(Range(i*2+1,i*2+2),Range(6,9)) = -(*X_norm2).at<float>(0,i) * (*X_norm1).col(i).t();

        //cout<< A(Range(i*2,i*2+2),Range(0,9)) << endl;
    }
    return A;
}

Mat buildCoord(void)
{
    Mat coord = Mat(3, (int)cols*(int)rows, CV_32FC1);
    for (int i=0; i<(int)rows; i++) {
        for (int j=0; j<(int)cols; j++) {
            coord.at<float>(0,i*(int)cols + j) = j;
            coord.at<float>(1,i*(int)cols + j) = i;
            coord.at<float>(2,i*(int)cols + j) = 1;
        }
    }
    return coord;
}
