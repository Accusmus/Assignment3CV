#include "image_loader.h"

image_loader::image_loader()
{

}

image_loader::~image_loader()
{
    //dtor
}

static const string foldernames[] = {
    "res/part1/hand1_",
    "res/part2/hand2_",
    "res/part3/hand3_",
    "res/part4/hand4_",
    "res/part5/hand5_",
};

static const string handtypes[] = {
    "0*","1*","2*","3*","4*","5*","6*","7*","8*","9*",
};

//initialises and reads all files to vector of vectors of mats
void image_loader::readFiles(vector<vector<vector<Mat> > > &images){
    vector<String> filenames;

    for(size_t foldernum = 0; foldernum < 5; foldernum++){
        vector<vector<Mat> > fn;
        images.push_back(fn);

        for(size_t handtype = 0; handtype < 10; handtype++){
            vector<Mat> ht;
            images[foldernum].push_back(ht);

            string path = foldernames[foldernum] + handtypes[handtype];
            glob(path, filenames, true);

            for(size_t i = 0; i < filenames.size(); i++){
                Mat image;
                image = cv::imread(filenames[i], CV_8UC1);
                if(image.data == NULL){
                    cout << "Error: no such file" << endl;
                    exit(1);
                }
                images[foldernum][handtype].push_back(image);
                cout << "img: " << filenames[i] << endl;
            }
        }
        cout << "Folder part" << foldernum << " loaded successfully" << endl;
    }
}
