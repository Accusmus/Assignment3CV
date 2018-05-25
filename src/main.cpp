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
        Mat drawing;

        //create windows for displaying
        namedWindow("original", 0);
        namedWindow("contours", 0);

        image = imread(argv[1], CV_8UC1);


        string filename = "descriptor.txt";
        fourier_loader loader = fourier_loader();

        vector<float> fourier = loader.getSingleFourierDescriptor(image, drawing);
        for(size_t i = 0; i < fourier.size(); i++){
            cout << fourier[i] << " ";
        }
        cout << endl;

// -------Load all files and create a descriptor file
//        loader.readFiles();
//        loader.getFourierDescriptor();
//        loader.writeDescriptorToFile(filename);
//
//        vector<vector<vector<Mat> > > imgs;
//        vector<vector<vector<Mat> > > contourImgs;
//
//        loader.getImages(imgs);
//        loader.getContourImages(contourImgs);

        //Show Images
//        imshow("original", imgs[0][1][0]);
        imshow("contours", drawing);
        imshow("original", image);

        waitKey(0);
        return 0;

    //Camera Input
    }else
    {
        cout << "Info: Interperate camera input." << endl;
    }
    return 0;
}
