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

const float pano_rows = 1800;
const float pano_cols = 4800;

const float offset_rows = 300;
const float offset_cols = 1600;

#define TO_EXTENDED_COORD 1
#define TO_REFERED_COORD 0

float norm_ma[3][3] = {{2/cols,0,-1},{0,2/rows,-1},{0,0,1}};
Mat norm_matrix = Mat(3, 3, CV_32FC1,norm_ma);

/// Function header
void siftDetector( int, void* );
void goodMatches(Mat descriptor1, Mat descriptor2,std::vector<DMatch>* good_matches);
Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
Mat buildCoord(int x_min, int x_max, int y_min, int y_max);
Mat coordTransX2Xprime(Mat x,Mat H); // H X2Xprime H.inv() Xprime2X
Mat coordCalib(Mat x,bool flag);
Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
void affineTrans(Mat* s_image, Mat* out_img, Mat coord_source, Mat coord_out);
Mat linearBlend(Mat img1, Mat img2);

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
    
    vector<DMatch> good_matches;
    goodMatches(descriptor_1, descriptor_2,&good_matches);
    //Mat img_matches;

    Mat H =  getTrans(keypoints_1, keypoints_2, good_matches);
   // cout << "H is " << H << endl;
    
    float tmp[4][3] = {{0,0,1},{1600,0,1},{0,1200,1},{1600,1200,1}};
    Mat m_tmp(4,3,CV_32FC1,tmp);
    Mat board_coord = m_tmp.t();
    
    Mat H_board_coord = coordTransX2Xprime(board_coord, H); // Refered Coordinate
    H_board_coord = coordCalib(H_board_coord,TO_EXTENDED_COORD); //Extended Coordinate
   // cout << "H_board_coord" << H_board_coord << endl;
    
    double x_min,x_max,y_min,y_max;
    
    minMaxIdx(H_board_coord.row(0), &x_min, &x_max);
    minMaxIdx(H_board_coord.row(1), &y_min, &y_max);
    cout << "x min" << endl << x_min << "  " <<  x_max  << "  "<< y_min << "  " << y_max << endl;

    
    Mat out_img_coord = buildCoord(x_min, x_max, y_min,y_max);
    Mat img_coord = coordCalib(out_img_coord, TO_REFERED_COORD);
    
    cout << "out_img_coor "<< endl << out_img_coord(Range(0, 3),Range(0, 20)) << endl;
    img_coord = coordTransX2Xprime(img_coord, H.inv());
    
    Mat out_img1((int)pano_rows,(int)pano_cols,CV_8UC1,Scalar::all(0));
    Mat out_img2((int)pano_rows,(int)pano_cols,CV_8UC1,Scalar::all(0));

    affineTrans(&src_gray_1, &out_img1, img_coord, out_img_coord);
    src_gray_2.copyTo(out_img2(Range((int)offset_rows,(int)rows+(int)offset_rows),Range((int)offset_cols,(int)offset_cols+(int)cols)));
    
    
    Mat dst = linearBlend(out_img1, out_img2);
    
    
    
    resize(dst , dst, Size(dst.cols/4,out_img1.rows/4));
    namedWindow(matches_window,0);
    imshow( "Good Matches", dst);
    
}


/**
 * @function goodMatches
 * @find Mathces
 */
void goodMatches(Mat descriptor1, Mat descriptor2,std::vector<DMatch> *good_matches)
{
    //Feature Matching
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    double max_dist = 0; double min_dist = 100;

    //Caculate max and min distances between keypoints
    for (int i =0 ; i < descriptor1.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
                }

    //Draw Good Matches
    for( int i = 0; i < descriptor1.rows; i++ )
    { if( matches[i].distance <= max(3*min_dist, 0.02) )
    { (*good_matches).push_back( matches[i]); }
    }
}
/**
 * @function buildA
 * @build Matrix A for DLT algorithm
 */

Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    // Creat Coordinate Matrix
    Mat X_norm1;
    Mat X_norm2;
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
    
    X_norm1 = norm_matrix*X_img1;
    X_norm2 = norm_matrix*X_img2;
    
    // Creat 8*9 Matrix, using 4 points
    Mat A(2*(int)good_matches.size(),9,CV_32FC1,Scalar::all(0));
    for (int i=0; i<good_matches.size(); i++) {
        //cout <<(*X_norm1).col(i).t() << endl;
        A(Range(i*2,i*2+1),Range(3,6)) = -X_norm2.at<float>(2,i) * X_norm1.col(i).t();
        A(Range(i*2,i*2+1),Range(6,9)) = X_norm2.at<float>(1,i) * X_norm1.col(i).t();
        A(Range(i*2+1,i*2+2),Range(0,3)) = X_norm2.at<float>(2,i) * X_norm1.col(i).t();
        A(Range(i*2+1,i*2+2),Range(6,9)) = -X_norm2.at<float>(0,i) * X_norm1.col(i).t();

        //cout<< A(Range(i*2,i*2+2),Range(0,9)) << endl;
    }
    return A;
}

Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    Mat A = buildA(keypoint_1, keypoint_2, good_matches);
    SVD svd(A);
    Mat h = svd.vt.t().col(svd.vt.rows - 1);
    Mat H = h.t();
    H = H.reshape(1, 3);
    return H;
}

Mat buildCoord(int x_min, int x_max, int y_min, int y_max)
{
    int x_range = x_max - x_min;
    int y_range = y_max - y_min;
    Mat coord = Mat(3, x_range*y_range, CV_32FC1);
    for (int i=0; i<(int)y_range; i++) {
        for (int j=0; j<(int)x_range; j++) {
            coord.at<float>(0,i*x_range + j) = j + x_min;
            coord.at<float>(1,i*x_range + j) = i + y_min;
            coord.at<float>(2,i*x_range + j) = 1;
        }
    }
    return coord;
}

Mat coordTransX2Xprime(Mat x, Mat H)
{
    Mat x_prime(3,x.cols,CV_32FC1);
    x_prime = norm_matrix*x;//x_norm
    x_prime = H*x_prime;//h_x
    x_prime = norm_matrix.inv()*x_prime;
    for (int i=0; i<x_prime.cols; i++) {
        x_prime.col(i) /= x_prime.at<float>(2,i);
    }
    return x_prime;
}

Mat coordCalib(Mat x, bool flag)
{
    Mat x_new;
    x.copyTo(x_new);
    if (flag) {
        x_new.row(0) += offset_cols;
        x_new.row(1) += offset_rows;
    }else
    {
        x_new.row(0) -= offset_cols;
        x_new.row(1) -= offset_rows;
    }
        return x_new;
}

void affineTrans(Mat* s_image, Mat* out_img, Mat coord_source, Mat coord_out)
{
    for (int i=0; i<coord_out.cols; i++) {
        int x = coord_out.at<float>(0,i);
        int y = coord_out.at<float>(1,i);
        int x2 = coord_source.at<float>(0,i);
        int y2 = coord_source.at<float>(1,i);
        if ( x2 >=0 && x2<1600 && y2 >=0 && y2<1200 ) {
            (*out_img).at<uchar>(y,x) = (*s_image).at<uchar>(y2,x2);
        }
    }
}

Mat linearBlend(Mat img1, Mat img2)
{
    Mat mask(img1.size(),CV_8UC1);
    Mat mask2(img1.size(),CV_8UC1);
    cv::min(img1, img2, mask);
    mask2 = (mask != 0);
    mask2 /= 255;
    mask = (mask == 0);
    mask /= 255;
    mask *= 2;
    mask = mask + mask2;
    Mat out_img(img1.size(),CV_8UC1);
    double alpha = 0.5;
    double beta = 1 - alpha;
    addWeighted(img1, alpha, img2, beta, 0, out_img);
    out_img = out_img.mul(mask);
    
    return out_img;
}