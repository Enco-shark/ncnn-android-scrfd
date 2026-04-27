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

// 识别结果 - 用于线程安全
static std::vector<std::string> g_recognition_results;
static std::vector<float> g_recognition_similarities;
static std::vector<float> g_face_rect_x;
static std::vector<float> g_face_rect_y;
static std::vector<float> g_face_rect_w;
static std::vector<float> g_face_rect_h;
static ncnn::Mutex recognition_lock;

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

                // 清理旧的识别结果 - 每一帧都更新
                {
                    ncnn::MutexLockGuard g2(recognition_lock);
                    g_recognition_results.clear();
                    g_recognition_similarities.clear();
                    g_face_rect_x.clear();
                    g_face_rect_y.clear();
                    g_face_rect_w.clear();
                    g_face_rect_h.clear();
                }

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
                            if (face.landmark[j].x < 0 || face.landmark[j].y < 0 ||
                                face.landmark[j].x >= rgb.cols || face.landmark[j].y >= rgb.rows) {
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

                        // 识别人脸
                        auto results = g_recognizer->recognize(face_img, adjusted_landmarks, 0.5f);

                        // 保存识别结果 - 使用互斥锁保护
                        {
                            ncnn::MutexLockGuard g2(recognition_lock);
                            if (i >= g_recognition_results.size())
                            {
                                g_recognition_results.resize(i + 1);
                                g_recognition_similarities.resize(i + 1);
                                g_face_rect_x.resize(i + 1);
                                g_face_rect_y.resize(i + 1);
                                g_face_rect_w.resize(i + 1);
                                g_face_rect_h.resize(i + 1);
                            }

                            // 保存人脸框坐标（原始检测框坐标，用于 Java 端绘制中文）
                            g_face_rect_x[i] = face.rect.x;
                            g_face_rect_y[i] = face.rect.y;
                            g_face_rect_w[i] = face.rect.width;
                            g_face_rect_h[i] = face.rect.height;

                            if (!results.empty())
                            {
                                __android_log_print(ANDROID_LOG_INFO, "ncnn", "Face %zu recognized: %s (similarity=%.4f)",
                                                  i, results[0].name.c_str(), results[0].similarity);
                                g_recognition_results[i] = results[0].name;
                                g_recognition_similarities[i] = results[0].similarity;
                            }
                            else
                            {
                                g_recognition_results[i] = "";
                                g_recognition_similarities[i] = 0.0f;
                                __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "Face %zu: No recognition match", i);
                            }
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

JNIEXPORT jboolean JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint modelid, jint cpugpu)
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
        int num_people = g_recognizer->get_num_people();
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Recognizer loaded successfully, num_people=%d", num_people);
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    if (g_camera) {
        g_camera->close();
        delete g_camera;
        g_camera = 0;
    }

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

JNIEXPORT jboolean JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    if (!g_camera)
        return JNI_FALSE;

    g_camera->close();
    delete g_camera;
    g_camera = 0;

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    if (!g_camera)
        return JNI_FALSE;

    g_camera->set_window(win);

    ANativeWindow_release(win);

    return JNI_TRUE;
}

JNIEXPORT jint JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getFaceCount(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(recognition_lock);
    return (jint)g_recognition_results.size();
}

JNIEXPORT jstring JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getRecognitionResult(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);

    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_results.size())
        return env->NewStringUTF("");

    return env->NewStringUTF(g_recognition_results[faceIndex].c_str());
}

JNIEXPORT jfloat JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getRecognitionSimilarity(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);

    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_similarities.size())
        return 0.0f;

    return g_recognition_similarities[faceIndex];
}

JNIEXPORT jfloat JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getFaceRectX(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);
    if (faceIndex < 0 || faceIndex >= (jint)g_face_rect_x.size()) return 0.0f;
    return g_face_rect_x[faceIndex];
}

JNIEXPORT jfloat JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getFaceRectY(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);
    if (faceIndex < 0 || faceIndex >= (jint)g_face_rect_y.size()) return 0.0f;
    return g_face_rect_y[faceIndex];
}

JNIEXPORT jfloat JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getFaceRectWidth(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);
    if (faceIndex < 0 || faceIndex >= (jint)g_face_rect_w.size()) return 0.0f;
    return g_face_rect_w[faceIndex];
}

JNIEXPORT jfloat JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_getFaceRectHeight(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);
    if (faceIndex < 0 || faceIndex >= (jint)g_face_rect_h.size()) return 0.0f;
    return g_face_rect_h[faceIndex];
}

JNIEXPORT void JNICALL Java_com_enco_shark_scrfdncnn_SCRFDNcnn_clearRecognitionResults(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(recognition_lock);
    g_recognition_results.clear();
    g_recognition_similarities.clear();
    g_face_rect_x.clear();
    g_face_rect_y.clear();
    g_face_rect_w.clear();
    g_face_rect_h.clear();
}

}
