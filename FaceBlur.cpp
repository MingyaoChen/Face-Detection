//
//  FaceBlur.cpp
//  opencv
//

#include "FaceRecognition.cpp"
//#include "SkinColorHistogram.cpp"
#include <iostream>
#include <fstream>
#include <vector>


class FaceBlur
{
private:

    FaceDetection *detector = NULL;
    FaceRecognition *recognizor = NULL;

    std::string model_folder = "/Users/Tenger/Desktop/model";

public:

    FaceBlur()
    {
        detector = new FaceDetection();
        recognizor = new FaceRecognition(detector);
    }



    void train()
    {
        //as C++ can not iterate files under a folder
        //read filename from a file

        //open file
        std::ifstream in;
        in.open("/Users/Tenger/Desktop/model/facename.txt");


        //format of file for training:
        //number of people
        //index number
        //filename1
        //filename2

        std::string filename;
        std::vector<cv::Mat> images;
        std::vector<int> labels;

        int count_person, label, count_face;

        in >> count_person;

        cv::Mat image;

        for( int i = 0; i < count_person; i ++ )
        {
            in >> label >> count_face;


            for( int j = 0; j < count_face; j ++ )
            {
                in >> filename;

                labels.push_back(label);

                image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
                cv::resize(image, image, cv::Size(1000, 1000));
                images.push_back(image);
            }
        }


        //train
        recognizor->train(images, labels);

        //close file
        in.close();
        images.clear();
        labels.clear();


        //save model
        recognizor->save(model_folder);



//        //set the recognizor threshold
//        std::ofstream out;
//
//        in.open("/Users/Tenger/Desktop/model/facename.txt");
//        out.open("/Users/Tenger/Desktop/model/threshold.txt");
//
//
//        std::vector<int> predict_result;
//        std::vector<double> predict_confidence;
//
//
//        in >> count_person;
//
//        for( int i = 0; i < count_person; i ++ )
//        {
//            in >> label >> count_face;
//
//            for( int j = 0; j < count_face; j ++ )
//            {
//                in >> filename;
//
//                if( j == count_face - 1 )
//                {
//                    recognizor->predictSingle(filename, predict_result, predict_confidence);
//
//                    out << filename << "    ";
//                    for( int predict_index = 0; predict_index < predict_result.size(); predict_index ++ )
//                    {
//                        out << predict_result[predict_index] << "  " << predict_confidence[predict_index] << "   ";
//                    }
//
//                    out << std::endl;
//
//                    predict_result.clear();
//                    predict_confidence.clear();
//                }
//            }
//
//        }
//
//
//        //close files
//        out.close();
//        in.close();
    }




//    void predict()
//    {
//        //load model
//        recognizor->load(model_folder);
//
//
//        //write into a file
//        std::ifstream in;
//        std::ofstream out;
//
//        in.open("/Users/Tenger/Desktop/model/filename.txt");
////        out.open("/Users/Tenger/Desktop/model/filename_output.txt");
//
//
//        std::string filename;
//        std::vector<int> predict_result;
//        std::vector<double> predict_confidence;
//
//        std::ostringstream outfilename;
//
//
//
//        std::vector<cv::Mat> faces;
//
//        recognizor->setThreshold(21600.0, 5650.0, 24.53);
//
//        while( std::getline(in, filename) )
//        {
//            detector->detect( filename, cv::Size(0, 0) );
//            detector->getFaces(faces);
//
//
//            for( cv::Mat face : faces )
//            {
//
//                cv::cvtColor(face, face, CV_BGR2GRAY);
//                recognizor->predictImage(face, predict_result, predict_confidence);
//
//
//                for( int i = 0; i <3; i ++ )
//                {
//
//                    outfilename << "/Users/Tenger/Desktop/model/images/";
//
//                    if( i == 0 )
//                    {
//                        //eigenfaces
//                        outfilename << "EIGENFACES/";
//                    }else if( i == 1 )
//                    {
//                        outfilename << "FISHERFACES/";
//                    }else if( i == 2 )
//                    {
//                        outfilename << "LBPH/";
//                    }
//
//
//                    if( predict_result[i] == -1 )
//                    {
//
//                        outfilename << "STRANGER/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//
//                    }else if( predict_result[i] == 1 )
//                    {
//                        outfilename << "BRIAN/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//                    }else if( predict_result[i] == 2 )
//                    {
//                        outfilename << "TENGQI/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//                    }else if( predict_result[i] == 3 )
//                    {
//                        outfilename << "STEFEN/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//                    }else if( predict_result[i] == 4 )
//                    {
//                        outfilename << "RAMI/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//                    }else if( predict_result[i] == 5 )
//                    {
//                        outfilename << "CATHAL/" << predict_confidence[0] << "_" << getfilename(filename);
//                        cv::imwrite(outfilename.str(), face);
//                    }
//
//                        outfilename.str("");
//                }
//
//
//
//
//                predict_result.clear();
//                predict_confidence.clear();
//            }
//
//
//            faces.clear();
//        }
//
//
//
//
//        //close file
////        out.close();
//        in.close();
//    }
//
//
//
//
//    std::string getfilename( std::string filename )
//    {
//        unsigned int found = filename.find_last_of('/');
//
//        return filename.substr(found + 1);
//    }
//
//
//    //mark faces on each image listed in a file
//    void markFaces()
//    {
//
//        //as C++ can not iterate files under a folder
//        //read filename from a file
//
//
//        //open file
//        std::ifstream in;
//        in.open("/Users/Tenger/Desktop/filename.txt");
//
//
//
//        std::string filename;
////        std::vector<int> detect_faces_count;
//
//
//        while( std::getline(in, filename) )
//        {
//
//            std::cout << filename << detector->detect(filename, cv::Size(0, 0)) << std::endl;
////            detect_faces_count.push_back( detector->detect(filename, cv::Size(0, 0)) );
//            detector->writeFacesOrigin(filename);
//
//        }
//
//
//        //close file
//        in.close();
//
//
////        //write file
////        std::ofstream out;
////        out.open("/Users/Tenger/Desktop/filename.txt");
////
////
////
////
////        //close file
////        out.close();
//
//    }
//
//

    //for the training data, extract faces and save
    void extractAndSave()
    {
        //read from facesname.txt
        std::ifstream in;
        in.open("/Volumes/YETENGQI/social_network/list.txt");


        //global variables
        int i = 25907;
        std::string infile, outfile;


        //read files in
        while( in >> infile >> outfile )
        {
//            in >> infile >> outfile;
            std::vector<cv::Mat> faces_image = detector->extract(infile, cv::Size(0, 0));

            //write face
            for( int j = 0; j < faces_image.size(); j ++ )
            {
                std::stringstream filename;

                filename << outfile << "/" << i << "_" << j << ".jpg";

                cv::imwrite(filename.str(), faces_image[j]);
            }
            i ++;
        }


        //close file
        in.close();
    }



    //print threshold on threshold.txt
    void getThreshold()
    {
        //read from facename.txt
        std::ifstream in;
        std::ofstream out;

        in.open("/Users/Tenger/Desktop/model/facename.txt");



        //global variables
        int count_person, count_face, label;

        //store information and train
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        std::vector<std::string> filenames;

        cv::Mat image;


        std::string filename;

        in >> count_person;


        for( int i = 0; i < count_person; i ++ )
        {
            in >> label >> count_face;


            for( int j = 0; j < count_face; j ++ )
            {
                in >> filename;

                filenames.push_back(filename);

                labels.push_back(label);

                image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
                cv::resize(image, image, cv::Size(1000, 1000));

                images.push_back(image);
            }
        }

        //close files

        in.close();



        //print confidences
        out.open("/Users/Tenger/Desktop/model/threshold.txt");

        //var
        std::vector<cv::Mat> images_var;
        std::vector<int> labels_var;

        std::vector<int> predict_result;
        std::vector<double> predict_confidence;

        for( int i = 0; i < images.size(); i ++ )
        {
            images_var = images;
            labels_var = labels;

            //remove target
            images_var.erase(images_var.begin() + i);
            labels_var.erase(labels_var.begin() + i);

            //train
            recognizor->train(images_var, labels_var);



            //predict
            recognizor->predictImage(images[i], predict_result, predict_confidence);


            //write on file
            out << filenames[i] << "    " << predict_result[0] << "--" << predict_confidence[0] << "    " << predict_result[1] << "--" << predict_confidence[1] << "    " << predict_result[2] << "--" << predict_confidence[2] << std::endl;


            predict_result.clear();
            predict_confidence.clear();
        }

        out.close();
    }

//    //print thresholds on threshold.txt
//    void getThreshold()
//    {
//        //read from facename.txt
//        std::ifstream in;
//        std::ofstream out;
//
//        in.open("/Users/Tenger/Desktop/model/facename.txt");
//        out.open("/Users/Tenger/Desktop/model/threshold.txt");
//
//        //global variables
//        int case_number, line_number;
//        std::string line;
//        std::vector<cv::Mat> faces_image;
//
//        in >> case_number;
//
//        for( int i = 0; i < case_number; i ++ )
//        {
//            in >> line_number >> line_number;
//            std::getline(in, line);
//
//
//            for( int j = 0; j < line_number; j ++ )
//            {
//                std::getline(in, line);
//
//                //detect
//                std::cout << "The number of faces detected is " << detector->detect(line, cv::Size(0, 0)) << std::endl;
//                detector->writeFacesOrigin(line);
//                //                detector->getFaces(faces_image);
//                //
//                //                cv::imwrite(line, faces_image[0]);
//
//                faces_image.clear();
//            }
//        }
//
//        //close file
//        out.close();
//        in.close();
//
//    }

	/*
    void skinCalc()
    {
        std::ifstream in;
        std::ofstream out;

        // read list file and write result to result.txt
        in.open("/Users/Tenger/Desktop/MMM2015/imagedata/list.txt");
        out.open("/Users/Tenger/Desktop/MMM2015/imagedata/result.txt");

        // global variables
        std::string filename;
        int var;
        SkinColorHistogram calculator;

        while( in >> filename >> var )
        {
            out << var << " " << calculator.calculate(filename) << std::endl;
        }

        out.close();
        in.close();
    }
    */

    //get the threshold(parameter) for the model
    //we assume that:

    ~FaceBlur()
    {
        delete detector;
        delete recognizor;
    }

    void blurOnOriginal()
    {
        const std::string &image_path = "/Users/Tenger/Desktop/20131125_105654_087.jpg";

        cv::Mat image = cv::imread(image_path);
        detector->blurFacesOrigin(image, cv::Size(0, 0));
    }
};





int main()
{


//    FaceBlur faceblur;
////    faceblur.extractAndSave();
////    faceblur.skinCalc();
//
////    faceblur.blurOnOriginal();
//    //train the faces
////    faceblur.train();
//
//
////    faceblur.predict();
////    faceblur.extractAndSave();
//    std::cout << CV_VERSION  << std::endl;
//
////    SkinColorHistogram sch;
////    std::cout << sch.calculate("/Volumes/YETENGQI/social_network/brian/workplace/faces/13924_0.jpg") << std::endl;
//
//    return 0;

//    FaceDetection detection;
//    cv::Mat image =  cv::imread("/Users/Tenger/Desktop/20131121_153847_101.jpg");
//
//    detection.writeFacesOrigin(image, cv::Size(0, 0));
////    cv::imwrite("/Users/Tenger/Desktop/20131121_174941_004.jpg", image);

    std::cout << CV_VERSION << std::endl;
}
