#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;


vector<vector<Point> > getContour(Mat &src);
void ellipticFourierDescriptors(vector<Point> &contour, vector<float> CE);


int main(int argc, char **argv)
{
    //Single Image
    if(argc == 2)
    {
        cout << "Info: interperate single image: " << argv[1] << endl;

        //Define image matrices
        Mat image;
        Mat drawing;

        //create windows for displaying
        namedWindow("original", 0);
        namedWindow("contours", 0);

        //read in image as arguement
        image = imread(argv[1], CV_8UC1);
        if(image.data == NULL){
            cout << "ERROR: Could not read file" << endl;
            return 1;
        }
        //Resize to be the recommended size
        resize(image, image, Size(640, 480), 0, 0, INTER_LINEAR);

        drawing = Mat::zeros(image.size(), CV_8UC3);

        vector<vector<Point> > contour = getContour(image);

        Scalar color = CV_RGB(0, 255, 0);
        drawContours(drawing, contour, 0, color, 1, 8);

        //Show Images
        imshow("original", image);
        imshow("contours", drawing);

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

void ellipticFourierDescriptors(vector<Point> &contour, vector<float> CE){
    //TODO

}
