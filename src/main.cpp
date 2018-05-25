#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include "fourier_loader.h"

using namespace std;
using namespace cv;

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

//        vector<vector<vector<Mat> > > images;
//        vector<vector<vector<Mat> > > drawings;
//        vector<vector<vector<vector<float> > > >fourier;

        string filename = "descriptor.txt";
        image_loader loader = image_loader();
        loader.readFiles();
        loader.getFourierDescriptor();
        loader.writeDescriptorToFile(filename);

        vector<vector<vector<Mat> > > imgs;
        vector<vector<vector<Mat> > > contourImgs;

        loader.getImages(imgs);
        loader.getContourImages(contourImgs);

        //Show Images
        imshow("original", imgs[0][1][0]);
        imshow("contours", contourImgs[0][1][0]);


        waitKey(0);
        return 0;

    //Camera Input
    }else
    {
        cout << "Info: Interperate camera input." << endl;
    }
    return 0;
}
