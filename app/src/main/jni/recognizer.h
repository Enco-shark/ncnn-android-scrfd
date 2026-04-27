#ifndef RECOGNIZER_H
#define RECOGNIZER_H

#include <android/asset_manager.h>
#include <opencv2/core/core.hpp>
#include <net.h>
#include <string>
#include <vector>

struct PersonResult
{
    std::string name;
    float similarity;
    int index;
};

class Recognizer
{
public:
    int load(AAssetManager* mgr, const char* model_type, bool use_gpu = false);
    int load_database(AAssetManager* mgr);

    std::vector<PersonResult> recognize(const cv::Mat& face_img,
                                        const std::vector<cv::Point2f>& landmarks,
                                        float threshold = 0.6f);
    int get_num_people() { return num_people; }

private:
    ncnn::Net arcface;
    std::vector<std::string> names;
    std::vector<std::vector<float>> embeddings;
    int num_people = 0;

    cv::Mat preprocess_face(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks);
    std::vector<float> extract_feature(const cv::Mat& face_img);
    float cosine_similarity(const std::vector<float>& feat1, const std::vector<float>& feat2);
};

#endif // RECOGNIZER_H