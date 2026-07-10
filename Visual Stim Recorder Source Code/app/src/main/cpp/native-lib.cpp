#include <jni.h>
#include <cstring>
#include <android/log.h>
#include "lsl_c.h"
#include <pthread.h>

static lsl_outlet outlet = NULL;
static pthread_mutex_t outlet_mutex = PTHREAD_MUTEX_INITIALIZER;

#define LOG_TAG "LSL"

extern "C" JNIEXPORT void JNICALL
Java_com_agili8_visualstimrecorder_MainActivity_initLSL(JNIEnv* env, jobject /* this */) {
    pthread_mutex_lock(&outlet_mutex);

    if (outlet != NULL) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "initLSL called but outlet already exists");
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    lsl_streaminfo info = lsl_create_streaminfo(
            "VisualStimRecorderLSLMarker", // Stream name
            "Markers",                     // Stream type
            1,                             // Number of channels
            LSL_IRREGULAR_RATE,            // Sampling rate
            cft_float32,                   // Channel format
            "LSLMarkerID"                  // Unique ID
    );

    if (!info) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to create streaminfo");
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    outlet = lsl_create_outlet(info, 0, 360);
    if (!outlet) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Failed to create outlet");
        lsl_destroy_streaminfo(info);
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    lsl_destroy_streaminfo(info); // Stream info is no longer needed after outlet creation
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "LSL outlet created successfully");
    pthread_mutex_unlock(&outlet_mutex);
}

extern "C" JNIEXPORT void JNICALL
Java_com_agili8_visualstimrecorder_MainActivity_sendMarker(JNIEnv* env, jobject /* this */) {
    pthread_mutex_lock(&outlet_mutex);

    if (!outlet) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "sendMarker called but outlet is null");
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    int32_t value[1] = { 1 };
//    int32_t ec = lsl_push_sample_i(outlet, value);
    int32_t ec = lsl_push_sample_it(outlet, value, lsl_local_clock());
    if (ec < 0) {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "Error in push_sample: %d", ec);
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Marker pushed: %f, Timestamp: %f", 1.0f, lsl_local_clock());
    pthread_mutex_unlock(&outlet_mutex);
}

extern "C" JNIEXPORT void JNICALL
Java_com_agili8_visualstimrecorder_MainActivity_stopLSL(JNIEnv* env, jobject /* this */) {
    pthread_mutex_lock(&outlet_mutex);

    if (!outlet) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "stopLSL called but outlet is null");
        pthread_mutex_unlock(&outlet_mutex);
        return;
    }

    lsl_destroy_outlet(outlet);
    outlet = NULL;
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "LSL outlet stopped and deleted");
    pthread_mutex_unlock(&outlet_mutex);
}