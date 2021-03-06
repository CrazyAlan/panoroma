#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp> 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "ceres_optimization.h"

using namespace cv;
using namespace std;

/// Global variables
cv::Mat src[4], src_gray[4];

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
#define REFEREE_ID 1
#define RANSAC_TIMES 34

float norm_ma[3][3] = {{2/cols,0,-1},{0,2/rows,-1},{0,0,1}};
cv::Mat norm_matrix = cv::Mat(3, 3, CV_32FC1,norm_ma);

float tmp[4][3] = {{0,0,1},{1600,0,1},{0,1200,1},{1600,1200,1}};
cv::Mat m_tmp(4,3,CV_32FC1,tmp);
cv::Mat board_coord = m_tmp.t();

/// Function header
void siftDetector( int, void* );
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch>* good_matches);
cv::Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
cv::Mat buildCoord(int x_min, int x_max, int y_min, int y_max);
cv::Mat coordTransX2Xprime(cv::Mat x,cv::Mat H); // H X2Xprime H.inv() Xprime2X
cv::Mat coordCalib(cv::Mat x,bool flag);
cv::Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
void affineTrans(cv::Mat* s_image, cv::Mat* out_img, cv::Mat coord_source, cv::Mat coord_out);
cv::Mat linearBlend(cv::Mat img1, cv::Mat img2);
void getImgNSrcCoord(cv::Mat H, cv::Mat *img_coord, cv::Mat *out_img_coord);
void ransac(vector<DMatch> good_matches, vector<DMatch>* inlier_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2);
void randomArray(int size, int range, int *array);
void buildX1X2ForReprojection(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> inlier_matches, Mate *X1, Mate *X2);
void matTransCVMat2Mate(cv::Mat *A, Mat3 *B, bool flag);


/**
 * @function main
 */

int main( int, char** argv )
{
    srand (time(NULL));
    
    /// Load source image and convert it to grayn
    for (int i=0; i<4; i++) {
        src[i] = imread( argv[1+i], 1 );
        cvtColor( src[i], src_gray[i], COLOR_BGR2GRAY );
    }
    
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
    vector<cv::KeyPoint> keypoints[4];
    cv::Mat descriptor[4];
    cv::Mat out_img[4];
    for (int i=0; i<4; i++) {
        detector.operator()(src_gray[i], cv::Mat(), keypoints[i],descriptor[i],false);
        out_img[i] = cv::Mat((int)pano_rows,(int)pano_cols,CV_8UC3,Scalar(0,0,0));
    }
    
    EstimateHomographyOptions options;
    options.expected_average_symmetric_distance = 0.0000002;
    

    
    cv::Mat dst;
    
    src[1].copyTo(out_img[1](Range((int)offset_rows,(int)rows+(int)offset_rows),Range((int)offset_cols,(int)offset_cols+(int)cols)));
    
    
    for (int i=0; i<3; i++) {
        //cv::Mat img_matches;
        if (i == REFEREE_ID) {
            continue;
        }
        vector<DMatch> good_matches;
        vector<DMatch> inlier_matches;
        goodMatches(descriptor[i], descriptor[REFEREE_ID],&good_matches);
        
        //Projective Transform Matrix
        //Transform From Img1 to Img2
        
        ransac(good_matches, &inlier_matches, keypoints[i], keypoints[REFEREE_ID]);
        cv::Mat H =  getTrans(keypoints[i], keypoints[REFEREE_ID], inlier_matches);
        
        //Using Reprojection Error to Optimize
        Mate X1(2,good_matches.size()), X2(2,good_matches.size());
        Mat3 H_hat;
        matTransCVMat2Mate(&H,&H_hat,1);
        buildX1X2ForReprojection(keypoints[i], keypoints[REFEREE_ID], good_matches, &X1, &X2);
        EstimateHomography2DFromReprojection(X1, X2,options,&H_hat);
        
        std::cout << "Original matrix:\n" << H << "\n";
        std::cout << "Estimated matrix:\n" << H_hat << "\n";

        
     //   matTransCVMat2Mate(&H, &H_hat, 0);
        cv::Mat img_coord;
        cv::Mat out_img_coord;
        getImgNSrcCoord(H, &img_coord, &out_img_coord);
        
        affineTrans(&src[i], &out_img[i], img_coord, out_img_coord);
        out_img[REFEREE_ID] = linearBlend(out_img[i], out_img[REFEREE_ID]);

    }
    
    
    resize(out_img[REFEREE_ID] , out_img[REFEREE_ID], Size(out_img[REFEREE_ID].cols/4,out_img[REFEREE_ID].rows/4));
    namedWindow(matches_window,0);
    imshow( "Good Matches", out_img[REFEREE_ID]);
    
}


/**
 * @function goodMatches
 * @find Mathces
 */
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch> *good_matches)
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
    { if( matches[i].distance <= max(4*min_dist, 0.02) )
    { (*good_matches).push_back( matches[i]); }
    }
}

/**
 * @function ransac
 * @choose largest number of inliers
 */
void ransac(vector<DMatch> good_matches, vector<DMatch>* inlier_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2)
{
    int max = 0;
    vector<DMatch> rnd_matches(4);
    for (int count=0; count< RANSAC_TIMES; count++) {
        int rnd_array[4];
        randomArray(4, (int)good_matches.size(), rnd_array);
        for (int i=0; i<4; i++) {
            rnd_matches[i] = good_matches[rnd_array[i]];
        }
        
        cv::Mat H_tmp = getTrans(keypoint_1, keypoint_2, rnd_matches);
        cv::Mat A = buildA(keypoint_1, keypoint_2, good_matches);
        cv::Mat A_H = A*(H_tmp.reshape(1,1)).t(); // 2n * 1
        A_H = A_H.mul(A_H);
        cv::Mat error((int)good_matches.size(),1,CV_32F);
        for (int i=0; i<good_matches.size(); i++) {
            error.at<float>(i,0) = A_H.at<float>(2*i, 0) + A_H.at<float>(2*i+1, 0);
        }
        
       // cout << "error are   " << endl << error << endl;
        
        double error_min, error_max;
        minMaxIdx(error.col(0), &error_min, &error_max);
        error = (error <= 1.50367e-03); // satisfied, return 255
        error /= 255;
        
        int inliers = sum(error)[0]; // channel zero
        
        if (inliers > max) {
            max = inliers;
            (*inlier_matches).clear();

            for (int i=0; i<good_matches.size(); i++) {
                if (error.at<uchar>(i,0) == 1) {
                    (*inlier_matches).push_back(good_matches[i]);
                }
            }
            
        }
        cout <<"Ransac Inliers  " <<  inliers << "Inliers total " <<(*inlier_matches).size() << endl;
        cout << "Rtio of inliers  " << float((*inlier_matches).size())/(float)good_matches.size() << endl;
    }
}
/**
 * @function buildA
 * @build Matrix A for DLT algorithm
 */

cv::Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    // Creat Coordinate Matrix
    cv::Mat X_norm1;
    cv::Mat X_norm2;
    cv::Mat X_img1(3,(int)good_matches.size(),CV_32FC1);
    cv::Mat X_img2(3,(int)good_matches.size(),CV_32FC1);
    for (int i=0; i<(int)good_matches.size(); i++) {
        int id1 = good_matches[i].queryIdx;
        int id2 = good_matches[i].trainIdx;
        float x[3] = {keypoint_1[id1].pt.x,keypoint_1[id1].pt.y,1};
        float x_prime[3] = {keypoint_2[id2].pt.x,keypoint_2[id2].pt.y,1};
        cv::Mat(3,1,CV_32FC1,x).copyTo(X_img1.col(i));
        cv::Mat(3, 1, CV_32FC1, x_prime).copyTo(X_img2.col(i));
    }
    
    X_norm1 = norm_matrix*X_img1;
    X_norm2 = norm_matrix*X_img2;
    
    // Creat 8*9 Matrix, using 4 points
    cv::Mat A(2*(int)good_matches.size(),9,CV_32FC1,Scalar::all(0));
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

void buildX1X2ForReprojection(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> inlier_matches, Mate *X1, Mate *X2)
{
    // Creat Coordinate Matrix
    cv::Mat X_norm1;
    cv::Mat X_norm2;
    cv::Mat X_img1(3,(int)inlier_matches.size(),CV_32FC1);
    cv::Mat X_img2(3,(int)inlier_matches.size(),CV_32FC1);
    for (int i=0; i<(int)inlier_matches.size(); i++) {
        int id1 = inlier_matches[i].queryIdx;
        int id2 = inlier_matches[i].trainIdx;
        float x[3] = {keypoint_1[id1].pt.x,keypoint_1[id1].pt.y,1};
        float x_prime[3] = {keypoint_2[id2].pt.x,keypoint_2[id2].pt.y,1};
        cv::Mat(3,1,CV_32FC1,x).copyTo(X_img1.col(i));
        cv::Mat(3, 1, CV_32FC1, x_prime).copyTo(X_img2.col(i));
    }
    
    X_norm1 = norm_matrix*X_img1;
    X_norm2 = norm_matrix*X_img2;
    
    for (int i=0; i<(int)inlier_matches.size(); i++) {
        (*X1)(0,i) = (double)X_norm1.at<float>(0,i);
        (*X1)(1,i) = (double)X_norm1.at<float>(1,i);
        (*X2)(0,i) = (double)X_norm2.at<float>(0,i);
        (*X2)(1,i) = (double)X_norm2.at<float>(1,i);
    }
}

void matTransCVMat2Mate(cv::Mat *A, Mat3 *B, bool flag)
{
    if(flag){ //flag=1, CV::MAT -> Mate
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                (*B)(i,j) = (double)(*A).at<float>(i,j);
            }
        }
    }
    else{
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                 (*A).at<float>(i,j) = (*B)(i,j);
            }
        }
    }
}


cv::Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    cv::Mat A = buildA(keypoint_1, keypoint_2, good_matches);
    SVD svd(A);
    cv::Mat h = svd.vt.t().col(svd.vt.rows - 1);
    cv::Mat H = h.t();
    H = H.reshape(1, 3);
    return H;
}

void getImgNSrcCoord(cv::Mat H, cv::Mat *img_coord, cv::Mat *out_img_coord)
{
    cv::Mat H_board_coord = coordTransX2Xprime(board_coord, H); // Refered Coordinate
    H_board_coord = coordCalib(H_board_coord,TO_EXTENDED_COORD); //Extended Coordinate
    
    double x_min,x_max,y_min,y_max;
    
    minMaxIdx(H_board_coord.row(0), &x_min, &x_max);
    minMaxIdx(H_board_coord.row(1), &y_min, &y_max);
    
    *out_img_coord = buildCoord(x_min, x_max, y_min,y_max);
    *img_coord = coordCalib(*out_img_coord, TO_REFERED_COORD);
    
    *img_coord = coordTransX2Xprime(*img_coord, H.inv());

}

cv::Mat buildCoord(int x_min, int x_max, int y_min, int y_max)
{
    int x_range = x_max - x_min;
    int y_range = y_max - y_min;
    cv::Mat coord = cv::Mat(3, x_range*y_range, CV_32FC1);
    for (int i=0; i<(int)y_range; i++) {
        for (int j=0; j<(int)x_range; j++) {
            coord.at<float>(0,i*x_range + j) = j + x_min;
            coord.at<float>(1,i*x_range + j) = i + y_min;
            coord.at<float>(2,i*x_range + j) = 1;
        }
    }
    return coord;
}

cv::Mat coordTransX2Xprime(cv::Mat x, cv::Mat H)
{
    cv::Mat x_prime(3,x.cols,CV_32FC1);
    x_prime = norm_matrix*x;//x_norm
    x_prime = H*x_prime;//h_x
    x_prime = norm_matrix.inv()*x_prime;
    for (int i=0; i<x_prime.cols; i++) {
        x_prime.col(i) /= x_prime.at<float>(2,i);
    }
    return x_prime;
}

cv::Mat coordCalib(cv::Mat x, bool flag)
{
    cv::Mat x_new;
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

void affineTrans(cv::Mat* s_image, cv::Mat* out_img, cv::Mat coord_source, cv::Mat coord_out)
{
    for (int i=0; i<coord_out.cols; i++) {
        int x = coord_out.at<float>(0,i);
        int y = coord_out.at<float>(1,i);
        int x2 = coord_source.at<float>(0,i);
        int y2 = coord_source.at<float>(1,i);
        if ( x2 >=0 && x2<1600 && y2 >=0 && y2<1200 ) {
            (*out_img).at<Vec3b>(y,x) = (*s_image).at<Vec3b>(y2,x2);
        }
    }
}

cv::Mat linearBlend(cv::Mat img1, cv::Mat img2)
{
    cv::Mat mask(img1.size(),CV_8UC1);
    cv::Mat mask2(img1.size(),CV_8UC1);
    cv::min(img1, img2, mask);
    mask2 = (mask != 0);
    mask2 /= 255;
    mask = (mask == 0);
    mask /= 255;
    mask *= 2;
    mask = mask + mask2;
    cv::Mat out_img(img1.size(),CV_8UC1);
    double alpha = 0.5;
    double beta = 1 - alpha;
    addWeighted(img1, alpha, img2, beta, 0, out_img);
    out_img = out_img.mul(mask);
    
    return out_img;
}

void randomArray(int size, int range, int *array)
{
    for (int i=0; i<size; i++) {
        array[i] = rand()%range;
        for (int y=0; y<i; y++) {
            if (array[i] == array[y]) {
                i--;
                break;
            }
        }
        
    }
}


