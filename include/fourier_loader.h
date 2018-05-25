#ifndef FOURIER_LOADER_H
#define FOURIER_LOADER_H

#include<vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>


#define PI 3.14159265359

using namespace std;
using namespace cv;


class fourier_loader
{
    public:
        fourier_loader();
        virtual ~fourier_loader();

        void readFiles();
        void getFourierDescriptor();
        void writeDescriptorToFile(string filename);

        void getImages(vector<vector<vector<Mat> > > &imgs);
        void getContourImages(vector<vector<vector<Mat> > > &contourImg);

    protected:

    private:
        vector<vector<vector<Mat> > > images;
        vector<vector<vector<Mat> > > drawings;
        vector<vector<vector<vector<float> > > >fourier;

        vector<vector<Point> > getContour(Mat &src);
        void ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE);
};

#endif // fOURIER_LOADER_H
