#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include "recognizer.h"
#include <android/asset_manager_jni.h>
#include <algorithm>
#include <cmath>
#include "cpu.h"

static const int ARCFACE_INPUT_SIZE = 112;

int Recognizer::load(AAssetManager* mgr, const char* model_type, bool use_gpu)
{
    arcface.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    arcface.opt = ncnn::Option();

#if NCNN_VULKAN
    arcface.opt.use_vulkan_compute = use_gpu;
#endif

    arcface.opt.num_threads = ncnn::get_big_cpu_count();

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", model_type);
    sprintf(modelpath, "%s.bin", model_type);

    int ret1 = arcface.load_param(mgr, parampath);
    int ret2 = arcface.load_model(mgr, modelpath);

    if (ret1 != 0 || ret2 != 0) {
        return -1;
    }

    return 0;
}

int Recognizer::load_database(AAssetManager* mgr)
{
    AAsset* asset = AAssetManager_open(mgr, "face_database.bin", AASSET_MODE_BUFFER);
    if (!asset) {
        return -1;
    }

    const void* data = AAsset_getBuffer(asset);
    off_t size = AAsset_getLength(asset);

    const char* ptr = static_cast<const char*>(data);

    num_people = *reinterpret_cast<const int32_t*>(ptr);
    ptr += 4;

    names.resize(num_people);

    for (int i = 0; i < num_people; i++) {
        int32_t name_len = *reinterpret_cast<const int32_t*>(ptr);
        ptr += 4;
        names[i] = std::string(ptr, ptr + name_len);
        ptr += name_len;
    }

    const int embedding_dim = 512;
    embeddings.resize(num_people, std::vector<float>(embedding_dim));

    const float* emb_ptr = reinterpret_cast<const float*>(ptr);
    for (int i = 0; i < num_people; i++) {
        std::copy(emb_ptr, emb_ptr + embedding_dim, embeddings[i].begin());
        emb_ptr += embedding_dim;
    }

    AAsset_close(asset);

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "Loaded %d faces from database", num_people);
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "First few names: %s, %s, %s",
                       num_people > 0 ? names[0].c_str() : "none",
                       num_people > 1 ? names[1].c_str() : "none",
                       num_people > 2 ? names[2].c_str() : "none");

    return 0;
}

cv::Mat Recognizer::preprocess_face(const cv::Mat& img, const std::vector<cv::Point2f>& landmarks)
{
    const cv::Point2f dst_pts[3] = {
            cv::Point2f(38.2946f, 51.6963f),
            cv::Point2f(73.5318f, 51.5014f),
            cv::Point2f(56.0252f, 71.7366f)
    };

    cv::Point2f src_pts[3] = {
            landmarks[0],
            landmarks[1],
            landmarks[2]
    };

    cv::Mat M = cv::getAffineTransform(src_pts, dst_pts);

    if (M.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "getAffineTransform failed");
        return cv::Mat();
    }

    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE));

    return aligned;
}

std::vector<float> Recognizer::extract_feature(const cv::Mat& face_img)
{
    if (face_img.empty()) {
        return std::vector<float>(512, 0.0f);
    }

    // face_img 来自 rgb(face_rect)，已经是 RGB 格式，直接使用 PIXEL_RGB
    ncnn::Mat in = ncnn::Mat::from_pixels(face_img.data, ncnn::Mat::PIXEL_RGB,
                                          ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1/128.f, 1/128.f, 1/128.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = arcface.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc1", out);

    int feat_dim = out.w;
    std::vector<float> feature(feat_dim);
    for (int i = 0; i < feat_dim; i++) {
        feature[i] = out[i];
    }

    float norm = 0.0f;
    for (int i = 0; i < feat_dim; i++) {
        norm += feature[i] * feature[i];
    }
    norm = sqrt(norm);

    if (norm > 0) {
        for (int i = 0; i < feat_dim; i++) {
            feature[i] /= norm;
        }
    }

    return feature;
}

float Recognizer::cosine_similarity(const std::vector<float>& feat1, const std::vector<float>& feat2)
{
    if (feat1.size() != feat2.size() || feat1.empty()) {
        return 0.0f;
    }
    float dot = 0.0f;
    for (size_t i = 0; i < feat1.size(); i++) {
        dot += feat1[i] * feat2[i];
    }
    return dot;
}

std::vector<PersonResult> Recognizer::recognize(const cv::Mat& face_img,
                                                const std::vector<cv::Point2f>& landmarks,
                                                float threshold)
{
    cv::Mat aligned = preprocess_face(face_img, landmarks);
    if (aligned.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Face preprocessing failed");
        return std::vector<PersonResult>();
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Face preprocessing successful, aligned size: %dx%d", aligned.cols, aligned.rows);

    std::vector<float> query_feat = extract_feature(aligned);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Feature extraction completed, feature size: %zu", query_feat.size());

    std::vector<PersonResult> results;
    results.reserve(10);

    float max_sim = 0.0f;
    int max_idx = -1;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Starting similarity comparison with %d people", num_people);

    for (int i = 0; i < num_people; i++) {
        float sim = cosine_similarity(query_feat, embeddings[i]);
        if (sim > max_sim) {
            max_sim = sim;
            max_idx = i;
        }
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Best match: idx=%d, similarity=%.4f, threshold=%.4f", max_idx, max_sim, threshold);

    if (max_sim >= threshold && max_idx >= 0) {
        PersonResult result;
        result.name = names[max_idx];
        result.similarity = max_sim;
        result.index = max_idx;
        results.push_back(result);
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Recognition match: %s (similarity=%.4f)", result.name.c_str(), result.similarity);
    }

    return results;
}