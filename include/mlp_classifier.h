#ifndef MLP_CLASSIFIER_H
#define MLP_CLASSIFIER_H

#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>

#include<vector>
#include<iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;


class mlp_classifier
{
    public:
        mlp_classifier(string data, string save, string load);
        virtual ~mlp_classifier();

        float getClassifierResult(Mat &sample);
        bool build_mlp_classifier( const string& data_filename, const string& filename_to_save, const string& filename_to_load);
        float classifier_predict(string filename_to_load, Mat &sample);

    protected:

    private:
        string data_filename;
        string filename_to_save;
        string filename_to_load;
};

#endif // MLP_CLASSIFIER_H
