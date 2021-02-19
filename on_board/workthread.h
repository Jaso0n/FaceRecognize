#ifndef WORKTHREAD_H
#define WORKTHREAD_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <assert.h>
#include <math.h>
#include <signal.h>
#include <QCoreApplication>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <malloc.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <fcntl.h>

// Header files for DNNDK API
#include <dnndk/dnndk.h>

// DPU input & output Node name for DenseBox
#define NODE_INPUT "L0"
#define NODE_CONV "pixel_conv"
#define NODE_OUTPUT "bb_output"

#define IMAGE_SCALE (0.02)
#define CONFIDENCE_THRESHOLD (0.65)
#define IOU_THRESHOLD (0.3)

/*******ResNet50 Parameter**********/
/* 7.71 GOP MAdds for ResNet50 */
#define RESNET50_WORKLOAD (0.38828f)
/* DPU Kernel name for ResNet50 */
#define KRENEL_RESNET50 "resnet50"
/* Input Node for Kernel ResNet50 */
#define INPUT_NODE      "ConvNd_1"
/* Output Node for Kernel ResNet50 */
#define OUTPUT_NODE     "Addmm_1"

using namespace std;
using namespace std::chrono;
using namespace cv;

typedef pair<int, Mat> pairImage;


class PairComp {  // An auxiliary class for sort the image pair according to its
                  // index
    public:
    bool operator()(const pairImage &n1, const pairImage &n2) const {
        if (n1.first == n2.first) return n1.first > n2.first;
        return n1.first > n2.first;
    }
};


extern QMutex mtxQueueInput;                                               // mutex of input queue
extern QMutex mtxQueueShow;                                                // mutex of display queue
extern queue<pairImage> queueInput;                                       // input queue
extern priority_queue<pairImage, vector<pairImage>, PairComp> queueShow;  // display queue
extern atomic<int> idxShowImage;  // next frame index to be display
extern int name_id;
//extern constexpr int workerNum;


class WorkThread : public QThread
{
    Q_OBJECT
public:
    explicit WorkThread(QObject *parent = nullptr);
    void run();

    // Load DPU Kernel for DenseBox neural network
    DPUKernel *DenseBoxKernel = dpuLoadKernel("ultra96v2_DenseBox_360x640_2304");
    DPUKernel *ResNet50Kernel = dpuLoadKernel(KRENEL_RESNET50);

    /******* ResNet50 Tool Functions **************/
    void LoadWords(string const &path, vector<string> &kinds);
    pair<int, float> CPUCalcVectorAngle(float facelist[2][512], float *result);
    void DrawImage(Mat &frame, vector<float> face, float facelist[2][512], float* result);
    string ReadText(string filename, int line);
    void runResnet50(DPUTask *taskResnet50,Mat &frame,vector<Mat> FaceROI, vector<vector<float>> Facept);

    /******* DenseBox Tool Functions **************/
    vector<vector<float>> NMS(const vector<vector<float>> &box, float nms);
    void softmax_2(const vector<float> &input, vector<float> &output);
    void LoadVector(float *facelist);
    void runDenseBox(DPUTask *task,Mat img,vector<Mat> &FacROI,vector<vector<float>> &res);
    /******* MultiThread for Face Detection and Recognition **********/

signals:
    //emit current index of face after recognize face
    void recognize_bool(int);


public slots:

private:
    int face_index;

};

#endif // WORKTHREAD_H
