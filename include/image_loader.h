#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include<vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <string>

using namespace std;
using namespace cv;


class image_loader
{
    public:
        image_loader();
        virtual ~image_loader();

        void readFiles(vector<vector<vector<Mat> > > &images);

    protected:

    private:
};

#endif // IMAGE_LOADER_H
