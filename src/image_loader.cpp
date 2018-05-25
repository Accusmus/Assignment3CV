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
void image_loader::readFiles(){
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

void image_loader::getFourierDescriptor(){
    //takes images and returns fourier descriptors
    for(size_t i = 0; i < images.size(); i++){ // for each hand num (folder)
        vector<vector<Mat> > folder;
        drawings.push_back(folder);
        vector<vector<vector<float> > > ceFolder;
        fourier.push_back(ceFolder);

        for(size_t j = 0; j < images[i].size(); j++){ //for each hand sign
            vector<Mat> hand;
            drawings[i].push_back(hand);
            vector<vector<float> >ceHand;
            fourier[i].push_back(ceHand);

            for(size_t k = 0; k < images[i][j].size(); k++){ //for each image
                resize(images[i][j][k], images[i][j][k], Size(640, 480), 0, 0, INTER_LINEAR);
                drawings[i][j].push_back(Mat::zeros(images[i][j][k].size(), CV_8UC3));


                vector<vector<Point> > contour = getContour(images[i][j][k]);

                Scalar color = CV_RGB(0, 255, 0);
                drawContours(drawings[i][j][k], contour, 0, color, 1, 8);

                vector<float> ceImage;
                fourier[i][j].push_back(ceImage);
                ellipticFourierDescriptors(contour[0], fourier[i][j][k]);
            }
        }
        cout << "Descriptors for hand " << i << " calculated" << endl;
    }
}

void image_loader::writeDescriptorToFile(string filename){
    ofstream myfile;
    myfile.open(filename);

    for(size_t folder = 0; folder < fourier.size(); folder++){
        for(size_t gesture = 0; gesture < fourier[folder].size(); gesture++){
            for(size_t imgNum = 0; imgNum < fourier[folder][gesture].size(); imgNum++){
                for(size_t dscrpt = 0; dscrpt < fourier[folder][gesture][imgNum].size(); dscrpt++){
                    cout << " size " << fourier[folder][gesture][imgNum].size() << endl;
                    if(dscrpt == 0){
                        myfile << gesture << ",";
                    }else if(dscrpt != fourier[folder][gesture][imgNum].size() - 1){
                        myfile << fourier[folder][gesture][imgNum][dscrpt] << ",";
                    }else{
                        myfile << fourier[folder][gesture][imgNum][dscrpt] << endl;
                    }
                }
            }
        }
    }

    myfile.close();
}

vector<vector<Point> > image_loader::getContour(Mat &src){
        //apply a median blur to smooth image
        medianBlur(src, src, 9);
        //apply threshold
        threshold(src, src, 5, 255, CV_THRESH_BINARY);
        //find contours

        vector<vector<Point> > contours;
        findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        int largestcontour = 0;
        long int largestsize = 0;

        for(int i = 0; i < contours.size(); i++){
            if(largestsize < contours[i].size()){
                largestsize = contours[i].size();
                largestcontour = i;
            }
        }
        vector<vector<Point> > ret;
        ret.push_back(contours[largestcontour]);
        return ret;
}

void image_loader::ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE){
    vector<float> ax, ay, bx, by;
    int m = contour.size();
    int n = 30;
    float t = (2*PI)/m;

    for(int k = 0; k < n; k++){
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);

        for(int i = 0; i < m; i++){
            ax[k] = ax[k] + contour[i].x * cos((k+1) * t * (i));
            bx[k] = bx[k] + contour[i].x * sin((k+1) * t * (i));
            ay[k] = ay[k] + contour[i].y * cos((k+1) * t * (i));
            by[k] = by[k] + contour[i].y * sin((k+1) * t * (i));
        }
        ax[k] = (ax[k]) / m;
        bx[k] = (bx[k]) / m;
        ay[k] = (ay[k]) / m;
        by[k] = (by[k]) / m;
    }
    for(int k = 0; k < n; k++){
        CE.push_back(sqrt((ax[k]*ax[k]+ay[k]*ay[k])/(ax[0]*ax[0]+ay[0]*ay[0]))+
                        sqrt((bx[k]*bx[k]+by[k]*by[k])/(bx[0]*bx[0]+by[0]*by[0])) );
    }
//    for(int count=0; count<n && count < CE.size(); count++){
//        cout << count << " CE " << CE[count] << " ax " << ax[count] << " ay " << ay[count] << " bx " << bx[count] << " by " << by[count] << endl;
//    }
}

void image_loader::getImages(vector<vector<vector<Mat> > > &imgs){
    imgs = images;
}

void image_loader::getContourImages(vector<vector<vector<Mat> > > &contourImg){
    contourImg = drawings;
}
