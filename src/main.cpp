#include <iostream>
#include <fstream>
#include <string>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include<ctime>
#include<chrono>

#include "fourier_loader.h"
#include "mlp_classifier.h"

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char **argv)
{
    //Single Image
    if(argc == 2)
    {
        //Define image matrices
        Mat src;
        Mat image;
        Mat drawing;

        //create windows for displaying
        namedWindow("original", 0);
        namedWindow("processed", 0);
        namedWindow("contours", 0);

        resizeWindow("original", Size(500,500));
        resizeWindow("processed", Size(500,500));
        resizeWindow("contours", Size(500,500));

        src = imread(argv[1], CV_8UC1);

        image = src;

        fourier_loader loader = fourier_loader();

        vector<float> fourier = loader.getSingleFourierDescriptor(image, drawing);

        Mat sample1 = (Mat_<float>(1,9) << fourier[1],fourier[2],fourier[3],fourier[4], fourier[5],fourier[6],fourier[7], fourier[8], fourier[9]);

        string data = "res/classifier/descriptor.txt";
        string save = "res/classifier/example.xml";
        string load = "res/classifier/example.xml";
        mlp_classifier classifier = mlp_classifier(data, save, load);
        float gesture = classifier.getClassifierResult(sample1);
        cout << "gesture: " << gesture << endl;

        char str[20];
        int gestureNum = (int)gesture;
        sprintf(str, "%d", gestureNum);
        putText(image, str, Point2f(280, 240), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255, 255), 4);


        imshow("contours", drawing);
        imshow("original", src);
        imshow("processed", image);

        waitKey(0);
        return 0;

    }else if(argc == 3){
        //create a descriptor file

        string filename = "descriptor.txt";

        fourier_loader loader = fourier_loader();

        //loader.readFiles();
        loader.getBulkFourierDescriptor();
        loader.writeDescriptorToFile(filename);

    //Camera Input
    }else{
        cout << "Info: Interperate camera input." << endl;

        Mat frame;          //Image from camera
        Mat originalImage;  //grey image from camera this is modified after median filter
        Mat processedImage; //resulting processed image
        Mat drawing;

        VideoCapture cap;
        cap.open(0);
        if (!cap.isOpened()){
            cout << "Failed to open camera" << endl;
            return 0;
        }

        cout << "Opened camera" << endl;

        // Create window to display images
        namedWindow("WebCam", 1);

        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        // read camera output to matrices
        cap >> frame;

        cvtColor(frame, originalImage, COLOR_RGB2GRAY);

        int key=0;

        double fps=0.0;
        while (1){
            system_clock::time_point start = system_clock::now();

            //get image
            cap >> frame;
            if( frame.empty()){
                break;
            }

            cvtColor(frame, originalImage, COLOR_RGB2GRAY);

            fourier_loader loader = fourier_loader();

            vector<float> fourier = loader.getSingleFourierDescriptor(originalImage, drawing);
            if(!fourier.empty()){
                Mat sample1 = (Mat_<float>(1,9) << fourier[1],fourier[2],fourier[3],fourier[4], fourier[5],fourier[6],fourier[7], fourier[8], fourier[9]);

                string data = "res/classifier/descriptor.txt";
                string save = "res/classifier/example.xml";
                string load = "res/classifier/example.xml";
                mlp_classifier classifier = mlp_classifier(data, save, load);
                float gesture = classifier.getClassifierResult(sample1);
                cout << "gesture: " << gesture << endl;
            }

            imshow("WebCam", originalImage);

            //if key is pressed then exit program
            key=waitKey(1);
            if(key==113 || key==27) return 0;//either esc or 'q'

            //finish of finding the time
            system_clock::time_point end = system_clock::now();
            double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            fps = 1000000/seconds;

            cout << "frames " << fps << " seconds " << seconds << endl;
        }

    }
    return 0;
}
