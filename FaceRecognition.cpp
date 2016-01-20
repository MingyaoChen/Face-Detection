//
//  FaceRecognition.cpp
//  opencv
//

#include "FaceDetection.cpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>


class FaceRecognition
{
private:

    cv::Ptr<cv::FaceRecognizer> eigenfaceRecognizor = NULL, fisherfaceRecognizor = NULL, LBPHRecognizor = NULL;


public:

    std::vector<cv::Mat> images;

    std::vector<int> labels;





    FaceRecognition( FaceDetection *detector )
    {
        eigenfaceRecognizor = cv::createEigenFaceRecognizer();
        fisherfaceRecognizor = cv::createFisherFaceRecognizer();
        LBPHRecognizor = cv::createLBPHFaceRecognizer();

    }




    void train( std::vector<cv::Mat> &images, std::vector<int> &labels )
    {

//        //turn all images to grey and same size
//        for( cv::Mat image : images )
//        {
//            cv::cvtColor(image, image, CV_BGR2GRAY);
//            cv::resize(image, image, cv::Size(500, 500));
//
//            cv::imshow("face", image);
//            cv::waitKey();
//        }


        eigenfaceRecognizor->train(images, labels);
        fisherfaceRecognizor->train(images, labels);
        LBPHRecognizor->train(images, labels);
    }




    void predict( std::string &image_path, std::vector<int> &predict_result, std::vector<double> &predict_confidence, double predicted_confidence = 0.0 )
    {

        std::vector<cv::Mat> faces_image;

        int predicted_label = -1;

        //make sure there is no previous result
        predict_result.clear();
        predict_confidence.clear();


        for( cv::Mat face : faces_image )
        {

            //turn it to grey image and adjust size
            cv::cvtColor(face, face, CV_BGR2GRAY);
            cv::resize(face, face, cv::Size(500, 500));


            //push results from all three methods to one vector

            //eigenfaces
            eigenfaceRecognizor->predict(face, predicted_label, predicted_confidence);
            predict_result.push_back(predicted_label);
            predict_confidence.push_back(predicted_confidence);


            //fisherfaces
            fisherfaceRecognizor->predict(face, predicted_label, predicted_confidence);
            predict_result.push_back(predicted_label);
            predict_confidence.push_back(predicted_confidence);



            //LBPH
            LBPHRecognizor->predict(face, predicted_label, predicted_confidence);
            predict_result.push_back(predicted_label);
            predict_confidence.push_back(predicted_confidence);

        }


        //blur stranger's face
//        detector->blurFacesOrigin(image_path);
    }


    void predictSingle( const std::string &image_path, std::vector<int> &predict_result, std::vector<double> &predict_confidence, double predicted_confidence = 0.0 )
    {

        //read image
        cv::Mat face = cv::imread(image_path);

        if( ! face.data )
        {
            std::cout << "Can not find image!" << std::endl;
        }

        //turn it to grey image and adjust size
        cv::cvtColor(face, face, CV_BGR2GRAY);
        cv::resize(face, face, cv::Size(500, 500));

        int predicted_label = -1;

        //push results from all three methods to one vector

        //eigenfaces
        eigenfaceRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);


        //fisherfaces
        fisherfaceRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);



        //LBPH
        LBPHRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);
    }



    // predict image in the memory
    void predictImage( cv::Mat &face, std::vector<int> &predict_result, std::vector<double> &predict_confidence, double predicted_confidence = 0.0 )
    {

        //resize image
        cv::resize(face, face, cv::Size(1000, 1000));

        int predicted_label = -1;

        //push results from all three methods to one vector

        //eigenfaces
        eigenfaceRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);


        //fisherfaces
        fisherfaceRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);



        //LBPH
        LBPHRecognizor->predict(face, predicted_label, predicted_confidence);
        predict_result.push_back(predicted_label);
        predict_confidence.push_back(predicted_confidence);
    }


    void setThreshold( double threshold1, double threshold2, double threshold3 )
    {
        eigenfaceRecognizor->set("threshold", threshold1);
        fisherfaceRecognizor->set("threshold", threshold2);
        LBPHRecognizor->set("threshold", threshold3);
    }


//    //changed
//    void predict( cv::Mat face, std::vector<int> &predict_result, std::vector<double> &predict_confidence, double predicted_confidence = 0.0 )
//    {
//
//
//        //
//        int predicted_label = -1;
//
//
//        //
//        predict_result.clear();
//        predict_confidence.clear();
//
//
//        //turn it to grey image and adjust size
//        cv::cvtColor(face, face, CV_BGR2GRAY);
//        cv::resize(face, face, cv::Size(500, 500));
//
//
//        //push results from all three methods to one vector
//
//        //eigenfaces
//        eigenfaceRecognizor->predict(face, predicted_label, predicted_confidence);
//        predict_result.push_back(predicted_label);
//        predict_confidence.push_back(predicted_confidence);
//
//
//        //fisherfaces
//        fisherfaceRecognizor->predict(face, predicted_label, predicted_confidence);
//        predict_result.push_back(predicted_label);
//        predict_confidence.push_back(predicted_confidence);
//
//
//
//        //LBPH
//        LBPHRecognizor->predict(face, predicted_label, predicted_confidence);
//        predict_result.push_back(predicted_label);
//        predict_confidence.push_back(predicted_confidence);
//
//
//
//
//    }




    void save( std::string model_path )
    {
        eigenfaceRecognizor->save( model_path + "/eigenface.xml");

        fisherfaceRecognizor->save( model_path + "/fisherface.xml");

        LBPHRecognizor->save( model_path + "/LBPH.xml");
    }




    void load( std::string model_path )
    {
        eigenfaceRecognizor->load( model_path + "/eigenface.xml");

        fisherfaceRecognizor->load( model_path + "/fisherface.xml");

        LBPHRecognizor->load( model_path + "/LBPH.xml");
    }

};
