#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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

        //read in image as arguement
        image = imread(argv[1], CV_8UC1);
        if(image.data == NULL){
            cout << "ERROR: Could not read file" << endl;
            return 1;
        }
        //Resize to be the recommended size
        resize(image, image, Size(640, 480), 0, 0, INTER_LINEAR);


        //Show Images
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
