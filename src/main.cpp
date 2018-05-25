#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "image_loader.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

#define PI 3.14159265359

vector<vector<Point> > getContour(Mat &src);
void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE);


int main(int argc, char **argv)
{
    //Single Image
    if(argc == 2)
    {
        cout << "Info: interperate single image: " << argv[1] << endl;

        //Define image matrices
        Mat image;

        //create windows for displaying
        namedWindow("original", 0);
        namedWindow("contours", 0);

        vector<vector<vector<Mat> > > images;
        vector<vector<vector<Mat> > > drawings;
        vector<vector<vector<vector<float> > > >fourier;

        image_loader loader = image_loader();
        loader.readFiles(images);

        for(size_t i = 0; i < images.size(); i++){ // for each hand num (folder)
            vector<vector<Mat> > folder;
            drawings.push_back(folder);
            vector<vector<vector<float> > > ceFolder;
            fourier.push_back(ceFolder);

            for(size_t j = 0; j < images[i].size(); j++){ //for each hand sign
                vector<Mat> hand;
                drawings[i].push_back(hand);
                vector<vector<float> >ceHand;
                fourier[i].push_back(ceHand);

                for(size_t k = 0; k < images[i][j].size(); k++){ //for each image
                    resize(images[i][j][k], images[i][j][k], Size(640, 480), 0, 0, INTER_LINEAR);
                    drawings[i][j].push_back(Mat::zeros(images[i][j][k].size(), CV_8UC3));


                    vector<vector<Point> > contour = getContour(images[i][j][k]);

                    Scalar color = CV_RGB(0, 255, 0);
                    drawContours(drawings[i][j][k], contour, 0, color, 1, 8);

                    vector<float> ceImage;
                    fourier[i][j].push_back(ceImage);
                    ellipticFourierDescriptors(contour[0], fourier[i][j][k]);
                }
            }
            cout << "Descriptors for hand " << i << " calculated" << endl;
        }

        ofstream myfile;
        myfile.open("descriptor.txt");
        //myfile << "writing to file\n" << endl;
        for(size_t folder = 0; folder < fourier.size(); folder++){
            for(size_t gesture = 0; gesture < fourier[folder].size(); gesture++){
                for(size_t imgNum = 0; imgNum < fourier[folder][gesture].size(); imgNum++){
                    for(size_t dscrpt = 0; dscrpt < fourier[folder][gesture][imgNum].size(); dscrpt++){
                        if(dscrpt == 0){
                            myfile << gesture << ",";
                        }else if(dscrpt != fourier[folder][gesture][imgNum].size() - 1){
                            myfile << fourier[folder][gesture][imgNum][dscrpt] << ",";
                        }else{
                            myfile << fourier[folder][gesture][imgNum][dscrpt] << endl;
                        }
                    }
                }
            }
        }
//        int folder = 0;
//        int gesture =0;
//        //int image = 0;
//        for(int z = 0; z < fourier.size(); z++){
//            for(int y = 0; y < fourier[z][gesture].size(); y++){
//                for(int x = 0; x < fourier[z][gesture][y].size(); x++){
//                    //cout << fourier[0][0].size() << endl;
//                    cout << fourier[z][gesture][y][x] << " ";
//                }
//                cout << endl;
//            }
//            cout << endl;
//        }

        myfile.close();
        //Show Images
        imshow("original", images[4][9][4]);
        imshow("contours", drawings[4][9][4]);


        waitKey(0);
        return 0;

    //Camera Input
    }else
    {
        cout << "Info: Interperate camera input." << endl;
    }
    return 0;
}

vector<vector<Point> > getContour(Mat &src){
        //apply a median blur to smooth image
        medianBlur(src, src, 9);
        //apply threshold
        threshold(src, src, 5, 255, CV_THRESH_BINARY);
        //find contours

        vector<vector<Point> > contours;
        findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        int largestcontour = 0;
        long int largestsize = 0;

        for(int i = 0; i < contours.size(); i++){
            if(largestsize < contours[i].size()){
                largestsize = contours[i].size();
                largestcontour = i;
            }
        }
        vector<vector<Point> > ret;
        ret.push_back(contours[largestcontour]);
        return ret;
}

void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE){
    vector<float> ax, ay, bx, by;
    int m = contour.size();
    int n = 10;
    float t = (2*PI)/m;

    for(int k = 0; k < n; k++){
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);

        for(int i = 0; i < m; i++){
            ax[k] = ax[k] + contour[i].x * cos((k+1) * t * (i));
            bx[k] = bx[k] + contour[i].x * sin((k+1) * t * (i));
            ay[k] = ay[k] + contour[i].y * cos((k+1) * t * (i));
            by[k] = by[k] + contour[i].y * sin((k+1) * t * (i));
        }
        ax[k] = (ax[k]) / m;
        bx[k] = (bx[k]) / m;
        ay[k] = (ay[k]) / m;
        by[k] = (by[k]) / m;
    }
    for(int k = 0; k < n; k++){
        CE.push_back(sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(ax[0]*ax[0]+ay[0]*ay[0]))+
                        sqrt((bx[k]*bx[k]+by[k]*by[k])/(bx[0]*bx[0]+by[0]*by[0])) );
    }
//    for(int count=0; count<n && count < CE.size(); count++){
//        cout << count << " CE " << CE[count] << " ax " << ax[count] << " ay " << ay[count] << " bx " << bx[count] << " by " << by[count] << endl;
//    }
}
