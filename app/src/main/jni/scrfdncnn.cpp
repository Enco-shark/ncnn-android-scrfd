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

// 添加传递识别结果的方法 - 用于线程安全
static std::vector<std::string> g_recognition_results;
static std::vector<float> g_recognition_similarities;
static ncnn::Mutex recognition_lock;  // 识别结果的专用互斥锁

// 用于中文绘制的 JNI 相关全局变量
static JavaVM* g_jvm = 0;
static jclass g_bitmap_class = 0;
static jmethodID g_bitmap_createBitmap = 0;
static jclass g_canvas_class = 0;
static jmethodID g_canvas_ctor = 0;
static jmethodID g_canvas_drawText = 0;
static jmethodID g_canvas_getWidth = 0;
static jmethodID g_canvas_getHeight = 0;
static jclass g_paint_class = 0;
static jmethodID g_paint_ctor = 0;
static jmethodID g_paint_setColor = 0;
static jmethodID g_paint_setTextSize = 0;
static jmethodID g_paint_setAntiAlias = 0;
static jmethodID g_paint_setTypeface = 0;
static jmethodID g_paint_measureText = 0;
static jclass g_typeface_class = 0;
static jmethodID g_typeface_default = 0;
static jmethodID g_typeface_create = 0;
static jclass g_bitmap_config_class = 0;
static jfieldID g_bitmap_config_argb8888 = 0;
static bool g_jni_initialized = false;

// 初始化 JNI 引用（在 JNI_OnLoad 中调用）
static bool init_jni_refs(JNIEnv* env)
{
    if (g_jni_initialized) return true;

    // Bitmap 类
    g_bitmap_class = (jclass)env->NewGlobalRef(env->FindClass("android/graphics/Bitmap"));
    if (!g_bitmap_class) return false;
    g_bitmap_createBitmap = env->GetStaticMethodID(g_bitmap_class, "createBitmap",
        "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!g_bitmap_createBitmap) return false;

    // Bitmap.Config 类
    g_bitmap_config_class = (jclass)env->NewGlobalRef(env->FindClass("android/graphics/Bitmap$Config"));
    if (!g_bitmap_config_class) return false;
    g_bitmap_config_argb8888 = env->GetStaticFieldID(g_bitmap_config_class, "ARGB_8888",
        "Landroid/graphics/Bitmap$Config;");

    // Canvas 类
    g_canvas_class = (jclass)env->NewGlobalRef(env->FindClass("android/graphics/Canvas"));
    if (!g_canvas_class) return false;
    g_canvas_ctor = env->GetMethodID(g_canvas_class, "<init>", "(Landroid/graphics/Bitmap;)V");
    g_canvas_drawText = env->GetMethodID(g_canvas_class, "drawText",
        "(Ljava/lang/String;FFLandroid/graphics/Paint;)V");
    g_canvas_getWidth = env->GetMethodID(g_canvas_class, "getWidth", "()I");
    g_canvas_getHeight = env->GetMethodID(g_canvas_class, "getHeight", "()I");

    // Paint 类
    g_paint_class = (jclass)env->NewGlobalRef(env->FindClass("android/graphics/Paint"));
    if (!g_paint_class) return false;
    g_paint_ctor = env->GetMethodID(g_paint_class, "<init>", "()V");
    g_paint_setColor = env->GetMethodID(g_paint_class, "setColor", "(I)V");
    g_paint_setTextSize = env->GetMethodID(g_paint_class, "setTextSize", "(F)V");
    g_paint_setAntiAlias = env->GetMethodID(g_paint_class, "setAntiAlias", "(Z)V");
    g_paint_setTypeface = env->GetMethodID(g_paint_class, "setTypeface",
        "(Landroid/graphics/Typeface;)V");
    g_paint_measureText = env->GetMethodID(g_paint_class, "measureText",
        "(Ljava/lang/String;)F");

    // Typeface 类
    g_typeface_class = (jclass)env->NewGlobalRef(env->FindClass("android/graphics/Typeface"));
    if (!g_typeface_class) return false;
    g_typeface_default = env->GetStaticMethodID(g_typeface_class, "defaultFromStyle",
        "(I)Landroid/graphics/Typeface;");
    g_typeface_create = env->GetStaticMethodID(g_typeface_class, "create",
        "(Ljava/lang/String;I)Landroid/graphics/Typeface;");

    g_jni_initialized = true;
    __android_log_print(ANDROID_LOG_INFO, "ncnn", "JNI refs initialized for Chinese text rendering");
    return true;
}

// 使用 Java Canvas 绘制中文文本到 cv::Mat 上
static void draw_chinese_text(cv::Mat& rgb, const std::string& text,
                               int x, int y, int color_r, int color_g, int color_b,
                               float font_size = 40.0f)
{
    if (!g_jvm || text.empty()) return;

    JNIEnv* env = 0;
    bool need_detach = false;

    // 获取当前线程的 JNIEnv
    int get_env_status = g_jvm->GetEnv((void**)&env, JNI_VERSION_1_4);
    if (get_env_status == JNI_EDETACHED) {
        if (g_jvm->AttachCurrentThread(&env, 0) != JNI_OK) {
            return;
        }
        need_detach = true;
    }

    if (!env || !g_jni_initialized) {
        if (need_detach) g_jvm->DetachCurrentThread();
        return;
    }

    // 测量文本宽度
    jstring jtext = env->NewStringUTF(text.c_str());
    if (!jtext) {
        if (need_detach) g_jvm->DetachCurrentThread();
        return;
    }

    // 创建 Paint 对象
    jobject paint = env->NewObject(g_paint_class, g_paint_ctor);
    env->CallVoidMethod(paint, g_paint_setAntiAlias, JNI_TRUE);
    env->CallVoidMethod(paint, g_paint_setTextSize, font_size);

    // 设置中文字体 - 使用 Typeface.DEFAULT（支持中文）
    jobject typeface = env->CallStaticObjectMethod(g_typeface_class, g_typeface_default, 0);
    env->CallVoidMethod(paint, g_paint_setTypeface, typeface);

    // 测量文本宽度
    jfloat text_width = env->CallFloatMethod(paint, g_paint_measureText, jtext);
    int text_height = (int)(font_size * 1.2f);  // 估算文本高度

    // 计算文本区域
    int text_x = x;
    int text_y = y;
    int bg_w = (int)text_width + 12;
    int bg_h = text_height + 8;

    // 确保不超出图像边界
    if (text_x + bg_w > rgb.cols) text_x = rgb.cols - bg_w - 5;
    if (text_x < 0) text_x = 0;
    if (text_y - bg_h < 0) text_y = bg_h;
    if (text_y > rgb.rows) text_y = rgb.rows - 5;

    // 创建 Bitmap（ARGB_8888）
    jobject config = env->GetStaticObjectField(g_bitmap_config_class, g_bitmap_config_argb8888);
    jobject bitmap = env->CallStaticObjectMethod(g_bitmap_class, g_bitmap_createBitmap,
        bg_w, bg_h, config);

    if (!bitmap) {
        env->DeleteLocalRef(jtext);
        env->DeleteLocalRef(paint);
        if (need_detach) g_jvm->DetachCurrentThread();
        return;
    }

    // 创建 Canvas
    jobject canvas = env->NewObject(g_canvas_class, g_canvas_ctor, bitmap);

    // 绘制白色背景
    jobject bg_paint = env->NewObject(g_paint_class, g_paint_ctor);
    env->CallVoidMethod(bg_paint, g_paint_setColor, 0xFFFFFFFF);  // 白色
    // 绘制背景矩形
    jclass j_rect_class = env->FindClass("android/graphics/Rect");
    jmethodID j_rect_ctor = env->GetMethodID(j_rect_class, "<init>", "(IIII)V");
    jobject rect = env->NewObject(j_rect_class, j_rect_ctor, 0, 0, bg_w, bg_h);
    jmethodID j_canvas_drawRect = env->GetMethodID(g_canvas_class, "drawRect",
        "(Landroid/graphics/Rect;Landroid/graphics/Paint;)V");
    env->CallVoidMethod(canvas, j_canvas_drawRect, rect, bg_paint);

    // 绘制黑色边框
    env->CallVoidMethod(bg_paint, g_paint_setColor, 0xFF000000);  // 黑色
    jmethodID j_canvas_drawRect2 = env->GetMethodID(g_canvas_class, "drawRect",
        "(FFFLandroid/graphics/Paint;)V");
    // 用 drawLine 画边框更简单，这里用 drawRect 画边框
    jmethodID j_paint_setStyle = env->GetMethodID(g_paint_class, "setStyle",
        "(Landroid/graphics/Paint$Style;)V");
    jclass j_paint_style_class = env->FindClass("android/graphics/Paint$Style");
    jfieldID j_paint_style_stroke = env->GetStaticFieldID(j_paint_style_class, "STROKE",
        "Landroid/graphics/Paint$Style;");
    jobject stroke_style = env->GetStaticObjectField(j_paint_style_class, j_paint_style_stroke);
    env->CallVoidMethod(bg_paint, j_paint_setStyle, stroke_style);
    jmethodID j_paint_setStrokeWidth = env->GetMethodID(g_paint_class, "setStrokeWidth", "(F)V");
    env->CallVoidMethod(bg_paint, j_paint_setStrokeWidth, 2.0f);
    env->CallVoidMethod(canvas, j_canvas_drawRect2, 0.0f, 0.0f, (float)bg_w, (float)bg_h, bg_paint);

    // 绘制中文文本（红色）
    int android_color = (0xFF << 24) | (color_r << 16) | (color_g << 8) | color_b;
    env->CallVoidMethod(paint, g_paint_setColor, android_color);
    env->CallVoidMethod(canvas, g_canvas_drawText, jtext, 6.0f, (float)(bg_h - 8), paint);

    // 从 Bitmap 获取像素数据
    jmethodID g_bitmap_getPixels = env->GetMethodID(g_bitmap_class, "getPixels",
        "([IIIIIII)V");
    jintArray pixels = env->NewIntArray(bg_w * bg_h);
    env->CallVoidMethod(bitmap, g_bitmap_getPixels, pixels, 0, bg_w, 0, 0, bg_w, bg_h);

    // 复制像素数据到 cv::Mat
    jint* pixel_data = env->GetIntArrayElements(pixels, 0);
    for (int row = 0; row < bg_h && (text_y - bg_h + row) < rgb.rows; row++) {
        int img_row = text_y - bg_h + row;
        if (img_row < 0) continue;
        for (int col = 0; col < bg_w && (text_x + col) < rgb.cols; col++) {
            int pixel = pixel_data[row * bg_w + col];
            int a = (pixel >> 24) & 0xFF;
            if (a > 0) {
                // 混合：Bitmap 是 ARGB，cv::Mat 是 BGR
                int b = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int r = pixel & 0xFF;
                cv::Vec3b& img_pixel = rgb.at<cv::Vec3b>(img_row, text_x + col);
                // Alpha 混合
                img_pixel[0] = (img_pixel[0] * (255 - a) + b * a) / 255;
                img_pixel[1] = (img_pixel[1] * (255 - a) + g * a) / 255;
                img_pixel[2] = (img_pixel[2] * (255 - a) + r * a) / 255;
            }
        }
    }

    env->ReleaseIntArrayElements(pixels, pixel_data, 0);

    // 清理
    env->DeleteLocalRef(pixels);
    env->DeleteLocalRef(rect);
    env->DeleteLocalRef(bg_paint);
    env->DeleteLocalRef(canvas);
    env->DeleteLocalRef(bitmap);
    env->DeleteLocalRef(config);
    env->DeleteLocalRef(typeface);
    env->DeleteLocalRef(paint);
    env->DeleteLocalRef(jtext);

    if (need_detach) g_jvm->DetachCurrentThread();
}

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

                // 清理旧的识别结果 - 每一帧都更新
                {
                    ncnn::MutexLockGuard g2(recognition_lock);
                    g_recognition_results.clear();
                    g_recognition_similarities.clear();
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

                        if (!results.empty())
                        {
                            __android_log_print(ANDROID_LOG_INFO, "ncnn", "Face %zu recognized: %s (similarity=%.4f)",
                                              i, results[0].name.c_str(), results[0].similarity);

                            // 使用中文绘制显示人名
                            int text_x = (int)face.rect.x;
                            int text_y = (int)face.rect.y - 5;

                            // 确保文本不超出图像边界
                            if (text_x < 0) text_x = 0;
                            if (text_y < 40) text_y = (int)face.rect.y + (int)face.rect.height + 40;

                            // 绘制中文人名（红色，字号36）
                            draw_chinese_text(rgb, results[0].name,
                                             text_x, text_y,
                                             255, 0, 0, 36.0f);

                            // 绘制相似度信息（使用 cv::putText，英文数字没问题）
                            char similarity_text[64];
                            sprintf(similarity_text, "%.1f%%", results[0].similarity * 100);
                            int sim_y = text_y + 5;
                            if (sim_y + 20 < rgb.rows)
                            {
                                cv::putText(rgb, similarity_text, cv::Point(text_x + 5, sim_y),
                                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);
                            }

                            // 保存识别结果 - 使用互斥锁保护
                            {
                                ncnn::MutexLockGuard g2(recognition_lock);
                                if (i >= g_recognition_results.size())
                                {
                                    g_recognition_results.resize(i + 1);
                                    g_recognition_similarities.resize(i + 1);
                                }
                                g_recognition_results[i] = results[0].name;
                                g_recognition_similarities[i] = results[0].similarity;
                            }
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

    g_jvm = vm;

    // 获取 JNIEnv 并初始化 JNI 引用
    JNIEnv* env = 0;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) == JNI_OK) {
        init_jni_refs(env);
    }

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    // 清理全局引用
    JNIEnv* env = 0;
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_4) == JNI_OK) {
        if (g_bitmap_class) env->DeleteGlobalRef(g_bitmap_class);
        if (g_canvas_class) env->DeleteGlobalRef(g_canvas_class);
        if (g_paint_class) env->DeleteGlobalRef(g_paint_class);
        if (g_typeface_class) env->DeleteGlobalRef(g_typeface_class);
        if (g_bitmap_config_class) env->DeleteGlobalRef(g_bitmap_config_class);
    }

    g_jvm = 0;
    g_jni_initialized = false;
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
        int num_people = g_recognizer->get_num_people();
        __android_log_print(ANDROID_LOG_INFO, "ncnn", "Recognizer loaded successfully, num_people=%d", num_people);
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    // 防止内存泄漏：如果已有旧实例，先释放
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


JNIEXPORT jstring JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getRecognitionResult(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);

    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_results.size())
        return env->NewStringUTF("");

    return env->NewStringUTF(g_recognition_results[faceIndex].c_str());
}

JNIEXPORT jfloat JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_getRecognitionSimilarity(JNIEnv* env, jobject thiz, jint faceIndex)
{
    ncnn::MutexLockGuard g(recognition_lock);

    if (faceIndex < 0 || faceIndex >= (jint)g_recognition_similarities.size())
        return 0.0f;

    return g_recognition_similarities[faceIndex];
}

JNIEXPORT void JNICALL Java_com_tencent_scrfdncnn_SCRFDNcnn_clearRecognitionResults(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(recognition_lock);
    g_recognition_results.clear();
    g_recognition_similarities.clear();
}

}
