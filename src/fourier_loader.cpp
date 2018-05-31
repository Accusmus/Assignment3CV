#include "fourier_loader.h"

fourier_loader::fourier_loader()
{

}

fourier_loader::~fourier_loader()
{
    //dtor
}

//all of the names of the image folders to load
static const vector<string> foldernames = {
    "res/part1/hand1_",
    "res/part2/hand2_",
    "res/part3/hand3_",
    "res/part4/hand4_",
    "res/part5/hand5_",
};

//all of the names of the image gestures types to load
static const vector<string> handtypes = {
    "0*","1*","2*","3*","4*","5*","6*","7*","8*","9*","a*","b*", "c*", "d*", "e*", "f*", "g*", "h*", "i*", "j*", "k*", "l*","m*","n*","o*","p*","q*", "r*", "s*", "t*", "u*", "v*", "w*", "x*", "y*", "z*"
};

//initialises and reads all files to vector of vectors of mats
void fourier_loader::readFiles(){
    //vector to hold all of the filenames
    vector<String> filenames;

    //loop through each of the folders
    for(size_t foldernum = 0; foldernum < foldernames.size(); foldernum++){
        vector<vector<Mat> > fn;
        images.push_back(fn);

        //loop through each gesture type
        for(size_t handtype = 0; handtype < handtypes.size(); handtype++){
            vector<Mat> ht;
            images[foldernum].push_back(ht);

            //use glob to find the filenames
            string path = foldernames[foldernum] + handtypes[handtype];
            glob(path, filenames, true);

            //loop through each image of the gesture in that folder
            for(size_t i = 0; i < filenames.size(); i++){
                Mat image;
                image = cv::imread(filenames[i], CV_8UC1);
                if(image.data == NULL){
                    cout << "Error: no such file" << endl;
                    exit(1);
                }
                //load images into memeory
                images[foldernum][handtype].push_back(image);
                cout << "img: " << filenames[i] << endl;
            }
        }
        cout << "Folder part:" << foldernames[foldernum] << " loaded successfully" << endl;
    }
}

vector<float> fourier_loader::getSingleFourierDescriptor(Mat &src, Mat &drawing, int thresh){
    resize(src, src, Size(640, 480), 0, 0, INTER_LINEAR);
    drawing = Mat::zeros(src.size(), CV_8UC3);
    //apply a median blur to smooth image
    medianBlur(src, src, 9);
    //apply threshold
    threshold(src, src, 20, 255, CV_THRESH_BINARY);

    vector<vector<Point> > contour = getContour(src);
    vector<float> fourier;
    if(!contour.empty()){
        Scalar color = CV_RGB(0, 255, 0);
        drawContours(drawing, contour, 0, color, 1, 8);
        ellipticFourierDescriptors(contour[0], fourier);
    }
    return fourier;
}

void fourier_loader::getBulkFourierDescriptor(){
    //takes images and returns fourier descriptors

    //for each folder
    for(size_t i = 0; i < images.size(); i++){ // for each hand num (folder)
        vector<vector<Mat> > folder;
        drawings.push_back(folder);
        vector<vector<vector<float> > > ceFolder;
        fourier.push_back(ceFolder);

        //for each gesture
        for(size_t j = 0; j < images[i].size(); j++){
            vector<Mat> hand;
            drawings[i].push_back(hand);
            vector<vector<float> >ceHand;
            fourier[i].push_back(ceHand);

            //for each image
            for(size_t k = 0; k < images[i][j].size(); k++){ //for each image
                resize(images[i][j][k], images[i][j][k], Size(640, 480), 0, 0, INTER_LINEAR);
                //apply a median blur to smooth image
                medianBlur(images[i][j][k], images[i][j][k], 9);
                //apply threshold
                threshold(images[i][j][k], images[i][j][k], 5, 255, CV_THRESH_BINARY);
                //create a zeroed out matrix for contour images
                drawings[i][j].push_back(Mat::zeros(images[i][j][k].size(), CV_8UC3));


                vector<vector<Point> > contour = getContour(images[i][j][k]);

                Scalar color = CV_RGB(0, 255, 0);
                drawContours(drawings[i][j][k], contour, 0, color, 1, 8);

                vector<float> ceImage;
                fourier[i][j].push_back(ceImage);
                ellipticFourierDescriptors(contour[0], fourier[i][j][k]);
            }
            cout << "Descriptors for Gesture " << handtypes[j] << " in folder " << foldernames[i] << " calculated" << endl;
        }
    }
}

//write all of the descriptors to a file
//Note: can only be called after getBulkDescriptor function
void fourier_loader::writeDescriptorToFile(string filename){
    const char gestureTable1[] = {
        '0', '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    };

    if(images.size() == 0){
        cout << "Error: No images to process" << endl;
        exit(1);
    }

    //create file stream
    ofstream myfile;
    myfile.open(filename);

    //for each folder of images
    for(size_t folder = 0; folder < fourier.size(); folder++){
        //for each gesure in folder
        for(size_t gesture = 0; gesture < fourier[folder].size(); gesture++){
            // for each image in gesture
            for(size_t imgNum = 0; imgNum < fourier[folder][gesture].size(); imgNum++){
                for(size_t dscrpt = 0; dscrpt < fourier[folder][gesture][imgNum].size(); dscrpt++){
                    //write the descriptor for this image
                    if(dscrpt == 0){
                        myfile << gestureTable1[gesture] << ",";
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

vector<vector<Point> > fourier_loader::getContour(Mat &src){
        //find contours
        vector<vector<Point> > contours;
        findContours(src, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        vector<vector<Point> > ret;
        if(!contours.empty()){
            int largestcontour = 0;
            long int largestsize = 0;

            for(size_t i = 0; i < contours.size(); i++){
                if(largestsize < contours[i].size()){
                    largestsize = contours[i].size();
                    largestcontour = i;
                }
            }
            ret.push_back(contours[largestcontour]);
        }
        return ret;
}

void fourier_loader::ellipticFourierDescriptors(vector<Point> &contour, vector<float> &CE){
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
}

void fourier_loader::getImages(vector<vector<vector<Mat> > > &imgs){
    imgs = images;
}

void fourier_loader::getContourImages(vector<vector<vector<Mat> > > &contourImg){
    contourImg = drawings;
}

vector<vector<vector<vector<float> > > > fourier_loader::getAllFourierDescriptors(){
    return fourier;
}
