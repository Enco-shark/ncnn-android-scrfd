#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "scrfd.h"
#include "recognizer.h"
#include "ndkcamera.h"

static SCRFD* g_scrfd = 0;
static Recognizer* g_recognizer = 0;
static ncnn::Mutex lock;

// 画"不支持"提示
static void draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);
    int x = (rgb.cols - label_size.width) / 2;
    int y = (rgb.rows - label_size.height) / 2;
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y - label_size.height),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 1);
}

// 画帧率
static void draw_fps(cv::Mat& rgb)
{
    // 简单静态帧率显示
    static double fps = 0;
    static int64 tick_count = 0;
    static int frame_count = 0;

    frame_count++;
    if (frame_count >= 30)
    {
        int64 now = cv::getTickCount();
        if (tick_count != 0)
        {
            fps = (double)frame_count / ((now - tick_count) / cv::getTickFrequency());
        }
        tick_count = now;
        frame_count = 0;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int x = 0;
    int y = label_size.height;
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y - label_size.height),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const
    {
        {
            ncnn::MutexLockGuard g(lock);

            if (g_scrfd)
            {
                std::vector<FaceObject> faceobjects;
                int ret = g_scrfd->detect(rgb, faceobjects);
                __android_log_print(ANDROID_LOG_INFO, "ncnn", "detect returned %d, found %zu faces", ret, faceobjects.size());

                g_scrfd->draw(rgb, faceobjects);

                // 单独处理人脸识别
                if (g_recognizer && !faceobjects.empty())
                {
                    __android_log_print(ANDROID_LOG_INFO, "ncnn", "Processing %zu faces for recognition", faceobjects.size());
                    for (size_t i = 0; i < faceobjects.size(); i++)
                    {
                        FaceObject& face = faceobjects[i];

                        // 检查人脸框的有效性
                        if (face.rect.width <= 0 || face.rect.height <= 0)
                        {
                            __android_log_print(ANDROID_LOG_WARN, "ncnn", "Face %zu has invalid rect", i);
                            continue;
                        }

                        // 检查landmarks有效性
                        bool has_valid_landmarks = true;
                        for (int j = 0; j < 5; j++) {
                            if (face.landmark[j].x <= 0 || face.landmark[j].y <= 0) {
                                has_valid_landmarks = false;
                                break;
                            }
                        }

                        if (!has_valid_landmarks)
                        {
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Face %zu has invalid landmarks", i);
                            continue;
                        }

                        // 使用原始检测框进行识别
                        cv::Rect face_rect = face.rect;
                        face_rect.x = std::max(0, face_rect.x - 10);
                        face_rect.y = std::max(0, face_rect.y - 10);
                        face_rect.width = std::min(rgb.cols - face_rect.x, face_rect.width + 20);
                        face_rect.height = std::min(rgb.rows - face_rect.y, face_rect.height + 20);

                        if (face_rect.width <= 0 || face_rect.height <= 0) {
                            __android_log_print(ANDROID_LOG_WARN, "ncnn", "Face %zu rect invalid after adjustment", i);
                            continue;
                        }

                        cv::Mat face_img = rgb(face_rect).clone();

                        std::vector<cv::Point2f> adjusted_landmarks(5);
                        for (int j = 0; j < 5; j++) {
                            adjusted_landmarks[j].x = face.landmark[j].x - face_rect.x;
                            adjusted_landmarks[j].y = face.landmark[j].y - face_rect.y;
                        }

                        // 识别人脸 - 降低阈值以提高识别率
                        auto results = g_recognizer->recognize(face_img, adjusted_landmarks, 0.3f);

                        if (!results.empty())
                        {
                            __android_log_print(ANDROID_LOG_INFO, "ncnn", "Face %zu recognized: %s (similarity=%.4f)",
                                              i, results[0].name.c_str(), results[0].similarity);

                            // 在人脸框旁边显示人名 - 尝试中文显示
                            std::string display_text = results[0].name;  // 直接使用原始人名

                            // 尝试使用UTF-8编码显示中文
                            int baseLine = 0;
                            cv::Size label_size = cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);

                            // 如果计算的文字尺寸异常，可能是中文显示问题
                            if (label_size.width <= 0 || label_size.height <= 0) {
                                __android_log_print(ANDROID_LOG_WARN, "ncnn", "Text size calculation failed for: %s", display_text.c_str());
                                // 回退到索引号显示
                                char index_text[32];
                                sprintf(index_text, "ID:%d", results[0].index);
                                display_text = std::string(index_text);
                                label_size = cv::getTextSize(display_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseLine);
                            }

                            // 在人脸框上方显示人名
                            int text_x = (int)face.rect.x;
                            int text_y = (int)face.rect.y - 10;

                            if (text_y < label_size.height)
                            {
                                text_y = (int)face.rect.y + (int)face.rect.height + label_size.height + 5;
                            }

                            // 防止超出图像边界
                            if (text_x + label_size.width > rgb.cols)
                            {
                                text_x = rgb.cols - label_size.width - 5;
                            }
                            if (text_y + label_size.height > rgb.rows)
                            {
                                text_y = rgb.rows - label_size.height - 5;
                            }

                            // 绘制白色背景框
                            cv::rectangle(rgb, cv::Rect(text_x - 3, text_y - label_size.height - baseLine,
                                                        label_size.width + 6, label_size.height + baseLine + 3),
                                         cv::Scalar(255, 255, 255), -1);

                            // 绘制黑色边框
                            cv::rectangle(rgb, cv::Rect(text_x - 3, text_y - label_size.height - baseLine,
                                                        label_size.width + 6, label_size.height + baseLine + 3),
                                         cv::Scalar(0, 0, 0), 2);

                            // 显示人名或索引号
                            cv::putText(rgb, display_text, cv::Point(text_x, text_y),
                                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

                            // 显示相似度信息
                            char similarity_text[64];
                            sprintf(similarity_text, "%.1f%%", results[0].similarity * 100);
                            cv::Size sim_size = cv::getTextSize(similarity_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

                            int sim_y = text_y + label_size.height + 15;
                            if (sim_y + sim_size.height < rgb.rows)
                            {
                                cv::putText(rgb, similarity_text, cv::Point(text_x, sim_y),
                                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
                            }

                            // 保存识别结果
                            if (i >= g_recognition_results.size())
                            {
                                g_recognition_results.resize(i + 1);
                                g_recognition_similarities.resize(i + 1);
                            }
                            g_recognition_results[i] = results[0].name;
                            g_recognition_similarities[i] = results[0].similarity;
                        }
                        else
                        {
                            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Face %zu: No recognition match (similarity below threshold)", i);
                        }
                    }
                }
                else if (!g_recognizer)
                {
                    __android_log_print(ANDROID_LOG_WARN, "ncnn", "Recognizer not initialized");
                }
            }
            else
            {
                __android_log_print(ANDROID_LOG_WARN, "ncnn", "g_scrfd is null");
                draw_unsupported(rgb);
            }
        }

        draw_fps(rgb);
    }
};

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
}

JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 7 || cpugpu < 0 || cpugpu > 1)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Invalid modelid %d or cpugpu %d", modelid, cpugpu);
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %d %d", modelid, cpugpu);

    const char* modeltypes[] =
            {
                    "500m",
                    "500m_kps",
                    "1g",
                    "2.5g",
                    "2.5g_kps",
                    "10g",
                    "10g_kps",
                    "34g",
            };

    const char* modeltype = modeltypes[modelid];
    bool use_gpu = cpugpu == 1;

    ncnn::MutexLockGuard g(lock);

    delete g_scrfd;
    g_scrfd = new SCRFD();

    int ret1 = g_scrfd->load(mgr, modeltype, use_gpu);

    if (ret1 != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "SCRFD model load failed with code %d, model: scrfd_%s", ret1, modeltype);
        delete g_scrfd;
        g_scrfd = 0;
        return JNI_FALSE;
    }

    __android_log_print(ANDROID_LOG_INFO, "ncnn", "SCRFD model loaded successfully");

    // 初始化人脸识别器
    delete g_recognizer;
    g_recognizer = new Recognizer();

    int ret2 = g_recognizer->load(mgr, "mobilefacenet", use_gpu);
    int ret3 = g_recognizer->load_database(mgr);

    if (ret2 != 0 || ret3 != 0)
    {
        __android_log_print(ANDROID_LOG_WARN, "ncnn", "Recognizer load failed (ret2=%d, ret3=%d), continuing without recognition", ret2, ret3);
        delete g_recognizer;
        g_recognizer = 0;
    }
    else
    {
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Recognizer loaded successfully");
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera = new MyNdkCamera();

    int ret = g_camera->open(facing);
    if (ret != 0)
    {
        delete g_camera;
        g_camera = 0;
        return JNI_FALSE;
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    if (!g_camera)
        return JNI_FALSE;

    g_camera->close();
    delete g_camera;
    g_camera = 0;

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    if (!g_camera)
        return JNI_FALSE;

    g_camera->set_window(win);

    ANativeWindow_release(win);

    return JNI_TRUE;
}

// 添加传递识别结果的方法
static std::vector<std::string> g_recognition_results;
static std::vector<float> g_recognition_similarities;

JNIEXPORT jstring JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getRecognitionResult(JNIEnv* env, jobject thiz, jint faceIndex)
{
    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_results.size())
        return env->NewStringUTF("");

    return env->NewStringUTF(g_recognition_results[faceIndex].c_str());
}

JNIEXPORT jfloat JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getRecognitionSimilarity(JNIEnv* env, jobject thiz, jint faceIndex)
{
    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_similarities.size())
        return 0.0f;

    return g_recognition_similarities[faceIndex];
}

JNIEXPORT void JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_clearRecognitionResults(JNIEnv* env, jobject thiz)
{
    g_recognition_results.clear();
    g_recognition_similarities.clear();
}

}