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
        Mat color;

        //create windows for displaying
        namedWindow("original", 0);
        namedWindow("processed", 0);
        namedWindow("contours", 0);
        namedWindow("result", 0);

        resizeWindow("original", Size(500,500));
        resizeWindow("processed", Size(500,500));
        resizeWindow("contours", Size(500,500));
        resizeWindow("result", Size(500, 500));

        src = imread(argv[1], CV_8UC1);
        if(!src.data){
            cout << "Error: File not found" << endl;
            exit(1);
        }

        color.create(src.size(), CV_8UC3);
        cvtColor(src, color, CV_GRAY2BGR);
        resize(color, color, Size(640,480),0,0, INTER_LINEAR);

        image = src;

        fourier_loader loader = fourier_loader();

        vector<float> fourier = loader.getSingleFourierDescriptor(image, drawing, 5);

        Mat sample1 = (Mat_<float>(1,29) << fourier[1],fourier[2],fourier[3],fourier[4],
                                                    fourier[5],fourier[6],fourier[7], fourier[8],
                                                    fourier[9],fourier[10],fourier[11],fourier[12],
                                                    fourier[13],fourier[14],fourier[15],fourier[16],
                                                    fourier[17],fourier[18],fourier[19],fourier[20],
                                                    fourier[21],fourier[22],fourier[23],fourier[24],
                                                    fourier[25],fourier[26],fourier[27],fourier[28],fourier[29]);

        string data = "res/classifier/descriptor2.txt";
        string save = "res/classifier/example2.xml";
        string load = "res/classifier/example2.xml";
        mlp_classifier classifier = mlp_classifier(data, save, load);
        float gesture = classifier.getClassifierResult(sample1);
        cout << "gesture: " << gesture << endl;

        char str[20];
        int gestureNum = (int)gesture;
        sprintf(str, "%d", gestureNum);
        putText(color, str, Point2f(570, 400), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255, 255), 4);


        imshow("contours", drawing);
        imshow("original", src);
        imshow("processed", image);
        imshow("result", color);

        waitKey(0);
        return 0;

    }else if(argc == 3){
        //read all training images
        //create a descriptor file
        cout << "create descriptor file" << endl;
        string filename = "descriptor2.txt";

        fourier_loader loader = fourier_loader();

        loader.readFiles();
        loader.getBulkFourierDescriptor();
        loader.writeDescriptorToFile(filename);

    //Camera Input
    }else{
        cout << "Info: Interperate camera input." << endl;

        Mat frame;          //Image from camera
        Mat processedImage; //resulting processed image
        Mat drawing;        //contour image

        VideoCapture cap;
        cap.open(0);
        if (!cap.isOpened()){
            cout << "Failed to open camera" << endl;
            return 0;
        }

        cout << "Opened camera" << endl;

        // Create window to display images
        namedWindow("Contour", 1);
        namedWindow("Processed", 1);

        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        // read camera output to matrices
        cap >> frame;

        cvtColor(frame, processedImage, COLOR_RGB2GRAY);

        int key=0;

        double fps=0.0;
        while (1){
            system_clock::time_point start = system_clock::now();

            //get image
            cap >> frame;
            if( frame.empty()){
                break;
            }

            cvtColor(frame, processedImage, COLOR_RGB2GRAY);

            fourier_loader loader = fourier_loader();

            vector<float> fourier = loader.getSingleFourierDescriptor(processedImage, drawing, 100);
            float gesture = -1;
            if(!fourier.empty()){
                //Mat sample1 = (Mat_<float>(1,9) << fourier[1],fourier[2],fourier[3],fourier[4], fourier[5],fourier[6],fourier[7], fourier[8], fourier[9]);
                Mat sample1 = (Mat_<float>(1,29) << fourier[1],fourier[2],fourier[3],fourier[4],
                                                    fourier[5],fourier[6],fourier[7], fourier[8],
                                                    fourier[9],fourier[10],fourier[11],fourier[12],
                                                    fourier[13],fourier[14],fourier[15],fourier[16],
                                                    fourier[17],fourier[18],fourier[19],fourier[20],
                                                    fourier[21],fourier[22],fourier[23],fourier[24],
                                                    fourier[25],fourier[26],fourier[27],fourier[28],fourier[29]);

                string data = "res/classifier/descriptor2.txt";
                string save = "res/classifier/example2.xml";
                string load = "res/classifier/example2.xml";
                mlp_classifier classifier = mlp_classifier(data, save, load);
                gesture = classifier.getClassifierResult(sample1);
                cout << "gesture: " << gesture << endl;
            }

            char str[20];
            int gestureNum = (int)gesture;
            sprintf(str, "%d", gestureNum);
            putText(drawing, str, Point2f(570, 400), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255, 255), 4);

            imshow("Contour", drawing);
            imshow("Processed", frame);

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
