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

//Assignment 3
//Matthew Crankshaw
//ID: 14303742

//to load single image just provide image path as arguement to program
//to load camera please provide no arguements to program and change camera settings so that hand is centure of attention in frame with dark background
//to test classifier provide 2 random arguements and set TEST_OR_CREATE to 0
//to create  a fourier descriptor provide 2 random arguements and set TEST_OR_CREATE to 1

//program is capable of testing all 5 gesture folders provided on stream

#define TEST_OR_CREATE 0

int main(int argc, char **argv)
{
    //gesture table for each type of gesture
    const char gestureTable[] = {
        '0', '1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
    };
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

        string data = "res/classifier/all.txt";
        string save = "res/classifier/all.xml";
        string load = "res/classifier/all.xml";
        mlp_classifier classifier = mlp_classifier(data, save, load);
        float gesture = classifier.getClassifierResult(sample1);
        if(gesture >= 10) gesture -= 7;
        cout << "gesture: " << gesture << endl;

        char str[20];
        int gestureNum = (int)gesture;
        sprintf(str, "%c", gestureTable[gestureNum]);
        putText(color, str, Point2f(570, 400), FONT_HERSHEY_PLAIN, 4, Scalar(0, 0, 255, 255), 4);


        imshow("contours", drawing);
        imshow("original", src);
        imshow("processed", image);
        imshow("result", color);

        waitKey(0);
        return 0;

    }else if(argc == 3){

//set to 1 to write a descriptor file
//set to 0 to test all the images
#if TEST_OR_CREATE
        //read all training images
        //create a descriptor file
        cout << "create descriptor file" << endl;
        string filename = "descriptor.txt";

        fourier_loader loader = fourier_loader();

        loader.readFiles();
        loader.getBulkFourierDescriptor();
        loader.writeDescriptorToFile(filename);

#else
        //read all files and get descriptor for all of them
        fourier_loader loader = fourier_loader();
        loader.readFiles();
        loader.getBulkFourierDescriptor();

        //load the classifier with xml file
        string data = "res/classifier/all.txt";
        string save = "res/classifier/all.xml";
        string load = "res/classifier/all.xml";
        mlp_classifier classifier = mlp_classifier(data, save, load);
        vector<vector<vector<vector<float> > > > fourier = loader.getAllFourierDescriptors();

        //set the counts to zero
        int countgesture[36], correctgesture[36];
        for(int i = 0; i < 36; i++) {
            countgesture[i] = 0;
            correctgesture[i] = 0;
        }
        //run through each image and check if it was predicted correctly
        for(size_t i = 0; i < fourier.size(); i++){
            for(size_t j = 0; j < fourier[i].size(); j++){
                for(size_t k = 0; k < fourier[i][j].size(); k++){
                    countgesture[j] ++;
                    Mat sample1 = (Mat_<float>(1,29) << fourier[i][j][k][1],fourier[i][j][k][2],fourier[i][j][k][3],fourier[i][j][k][4],
                                                    fourier[i][j][k][5],fourier[i][j][k][6],fourier[i][j][k][7], fourier[i][j][k][8],
                                                    fourier[i][j][k][9],fourier[i][j][k][10],fourier[i][j][k][11],fourier[i][j][k][12],
                                                    fourier[i][j][k][13],fourier[i][j][k][14],fourier[i][j][k][15],fourier[i][j][k][16],
                                                    fourier[i][j][k][17],fourier[i][j][k][18],fourier[i][j][k][19],fourier[i][j][k][20],
                                                    fourier[i][j][k][21],fourier[i][j][k][22],fourier[i][j][k][23],fourier[i][j][k][24],
                                                    fourier[i][j][k][25],fourier[i][j][k][26],fourier[i][j][k][27],fourier[i][j][k][28],fourier[i][j][k][29]);

                    float gesture = classifier.getClassifierResult(sample1);
                    if(gesture >= 10) gesture -= 7;
                    if(gesture == j){
                        correctgesture[j] ++;
                    }
                }
            }
        }
        //print out results
        for(int i = 0; i < 36; i++){
            cout << "Gesture " << gestureTable[i] << " correct: (" << correctgesture[i]  << "/" << countgesture[i];
            double percent = correctgesture[i] * 100 / countgesture[i];
            cout << ") Percent: " << percent << "\%" << endl;
        }
#endif

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

            //convert image to gray
            cvtColor(frame, processedImage, COLOR_RGB2GRAY);

            fourier_loader loader = fourier_loader();
            //load a single image and get the fourier descriptor for it
            vector<float> fourier = loader.getSingleFourierDescriptor(processedImage, drawing, 100);
            float gesture = -1;
            //make sure that if no fourier is found program doesnt crash
            if(!fourier.empty()){
                //Mat sample1 = (Mat_<float>(1,9) << fourier[1],fourier[2],fourier[3],fourier[4], fourier[5],fourier[6],fourier[7], fourier[8], fourier[9]);
                Mat sample1 = (Mat_<float>(1,29) << fourier[1],fourier[2],fourier[3],fourier[4],
                                                    fourier[5],fourier[6],fourier[7], fourier[8],
                                                    fourier[9],fourier[10],fourier[11],fourier[12],
                                                    fourier[13],fourier[14],fourier[15],fourier[16],
                                                    fourier[17],fourier[18],fourier[19],fourier[20],
                                                    fourier[21],fourier[22],fourier[23],fourier[24],
                                                    fourier[25],fourier[26],fourier[27],fourier[28],fourier[29]);

                string data = "res/classifier/all.txt";
                string save = "res/classifier/all.xml";
                string load = "res/classifier/all.xml";
                mlp_classifier classifier = mlp_classifier(data, save, load);
                gesture = classifier.getClassifierResult(sample1);
                if(gesture >= 10) gesture -= 7;
                cout << "gesture: " << gesture << endl;
            }

            char str[20];
            int gestureNum = (int)gesture;
            sprintf(str, "%c", gestureTable[gestureNum]);
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
