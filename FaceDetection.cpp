//
//  FaceDetection.cpp
//  opencv
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <string>

class FaceDetection
{
private:

    cv::CascadeClassifier classifier;//used to detect the object in a vedio stream

    std::string model_path = "/Users/Tenger/Documents/Workspace/JavaWorkspace/Jar/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_alt_tree.xml";

public:

    //construction of the class, load the classifier
    FaceDetection()
    {
        if( !classifier.load(model_path) )
        {
            std::cout << "Can not load training model!" << std::endl;
        }
    }


    std::vector<cv::Rect> detect(const std::string& image_path, const cv::Size& min_size)
    {
        //detected faces
        std::vector<cv::Rect> faces;


        //load image
        cv::Mat image = cv::imread(image_path);

        if( ! image.data )
        {
            std::cout << "Could not find images!" << std::endl;
        }

        classifier.detectMultiScale(image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, min_size);
//        std::cout << faces.size() << " faces have been detected!" << std::endl;

        return faces;
    }

    std::vector<cv::Rect> detect(cv::Mat& image, const cv::Size& min_size)
    {
        //detected faces
        std::vector<cv::Rect> faces;


        if( ! image.data )
        {
            std::cout << "Could not find images!" << std::endl;
        }

        classifier.detectMultiScale(image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, min_size );
        //        std::cout << faces.size() << " faces have been detected!" << std::endl;


        return faces;
    }


    std::vector<cv::Mat> extract(const std::string& image_path, const cv::Size& min_size)
    {
        //detected faces
        std::vector<cv::Rect> rectangles;


        //load image
        cv::Mat image = cv::imread(image_path);

        if( ! image.data )
        {
            std::cout << "Could not find images!" << std::endl;
        }

        classifier.detectMultiScale(image, rectangles, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, min_size );


        //copy faces
        std::vector<cv::Mat> faces;
        cv::Mat face;

        for( cv::Rect rect : rectangles )
        {
            image(rect).copyTo(face);
            faces.push_back(face);
        }

        return faces;
    }

    std::vector<cv::Mat> extract(cv::Mat &image, const cv::Size& min_size)
    {
        //detected faces
        std::vector<cv::Rect> rectangles;


        if( ! image.data )
        {
            std::cout << "Could not find images!" << std::endl;
        }

        classifier.detectMultiScale(image, rectangles, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, min_size );


        //copy faces
        std::vector<cv::Mat> faces;
        cv::Mat face;

        for( cv::Rect rect : rectangles )
        {
            image(rect).copyTo(face);
            faces.push_back(face);
        }

        return faces;
    }

    //draw a rectangle to the detected face rectangle
    void writeFacesOrigin( cv::Mat &image, const cv::Size& min_size )
    {
        std::vector<cv::Rect> faces = detect(image, min_size);

        for( int i = 0; i < faces.size(); i ++ )
        {
            cv::rectangle(image, cv::Point(faces[i].x, faces[i].y),
								cv::Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
								cv::Scalar(0, 255, 0), 4);
        }
    }


    //write detected faces and blur known faces on the images
    void blurFacesOrigin( cv::Mat &image, const cv::Size& min_size )
    {
        // detect and then blur
        std::vector<cv::Rect> faces = detect(image, min_size);

        for( int i = 0; i < faces.size(); i ++ )
        {
            cv::blur(image(faces[i]), image(faces[i]), cv::Size(100, 100));
        }

    }


};
