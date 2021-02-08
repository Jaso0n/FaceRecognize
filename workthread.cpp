#include "workthread.h"

QMutex mtxQueueInput;                                               // mutex of input queue
QMutex mtxQueueShow;                                                // mutex of display queue
queue<pairImage> queueInput;                                       // input queue
priority_queue<pairImage, vector<pairImage>, PairComp> queueShow;  // display queue
atomic<int> idxShowImage(0);  // next frame index to be display
//constexpr int workerNum = 2;

WorkThread::WorkThread(QObject *parent) : QThread(parent)
{
    face_index = -1;
}


void WorkThread::run()
{
    vector<Mat> FaceROI;//face region
    vector<vector<float> > Facept;//xmin,ymin,xmax,ymax,face region coordinate
    pair<int, Mat> pairIndexImage;
    // Create DPU Tasks from DPU Kernel
    DPUTask *detectfaceTask = dpuCreateTask(DenseBoxKernel,0);
    DPUTask *recogfaceTask = dpuCreateTask(ResNet50Kernel,0);
    while (true) {
        mtxQueueInput.lock();
        if (queueInput.empty()) {
            mtxQueueInput.unlock();
            continue;
        } else {
            // Get an image from input queue
            pairIndexImage = queueInput.front();
            queueInput.pop();
        }
        mtxQueueInput.unlock();
        // Process the image using DenseBox mode
        FaceROI.clear();
        Facept.clear();
        runDenseBox(detectfaceTask, pairIndexImage.second,FaceROI,Facept);
        if((FaceROI.size() > 0) && (Facept.size() > 0))
            runResnet50(recogfaceTask, pairIndexImage.second,FaceROI,Facept);
        mtxQueueShow.lock();
        // Put the processed iamge to show queue
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
    }
}


/**
 * @brief NMS - Discard overlapping boxes using NMS
 *
 * @param box - input box vector
 * @param nms - IOU threshold
 *
 * @ret - output box vector after discarding overlapping boxes
 */
vector<vector<float>> WorkThread::NMS(const vector<vector<float>> &box, float nms) {
    size_t count = box.size();
    vector<pair<size_t, float>> order(count);
    for (size_t i = 0; i < count; ++i) {
        order[i].first = i;
        order[i].second = box[i][4];
    }

    sort(order.begin(), order.end(), [](const pair<int, float> &ls, const pair<int, float> &rs) {
        return ls.second > rs.second;
    });

    vector<int> keep;
    vector<bool> exist_box(count, true);
    for (size_t _i = 0; _i < count; ++_i) {
        size_t i = order[_i].first;
        float x1, y1, x2, y2, w, h, iarea, jarea, inter, ovr;
        if (!exist_box[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < count; ++_j) {
            size_t j = order[_j].first;
            if (!exist_box[j]) continue;
            x1 = max(box[i][0], box[j][0]);
            y1 = max(box[i][1], box[j][1]);
            x2 = min(box[i][2], box[j][2]);
            y2 = min(box[i][3], box[j][3]);
            w = max(float(0.0), x2 - x1 + 1);
            h = max(float(0.0), y2 - y1 + 1);
            iarea = (box[i][2] - box[i][0] + 1) * (box[i][3] - box[i][1] + 1);
            jarea = (box[j][2] - box[j][0] + 1) * (box[j][3] - box[j][1] + 1);
            inter = w * h;
            ovr = inter / (iarea + jarea - inter);
            if (ovr >= nms) exist_box[j] = false;
        }
    }

    vector<vector<float>> result;
    result.reserve(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) {
        result.push_back(box[keep[i]]);
    }

    return result;
}


/**
 * @brief softmax_2 - 2-class softmax calculation
 *
 * @param input   - vector of input data
 * @param output  - output vecotr
 *
 * @return none
 */
void WorkThread::softmax_2(const vector<float> &input, vector<float> &output) {
    for (size_t n = 0; n < input.size(); n += 2) {
        float sum = 0;
        for (auto i = n; i < n + 2; i++) {
            output[i] = exp(input[i]);
            sum += output[i];
        }
        for (auto i = n; i < n + 2; i++) {
            output[i] /= sum;
        }
    }
}


void WorkThread:: LoadVector(float *facelist) {

    ifstream FaceVector;
    FaceVector.open("./faces.txt",ios::in);

    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 512; j++)
        {
            FaceVector >> *(facelist + i*512 + j);
        }
    }
    FaceVector.close();
}


/**
 * @brief runDenseBox - Run DPU Task for Densebox
 *
 * @param task - pointer to a DPU Task
 * @param img  - input image in OpenCV's Mat format
 *
 * @return none
 */
void WorkThread:: runDenseBox(DPUTask *task, Mat img, vector<Mat> &FaceROI, vector<vector<float> > &Facept) {

    DPUTensor *conv_in_tensor = dpuGetInputTensor(task, NODE_INPUT);// get the inputTensor 360*640*3
    int inHeight = dpuGetTensorHeight(conv_in_tensor);// input tensor height
    int inWidth = dpuGetTensorWidth(conv_in_tensor);//input tensor width

    float scale_w = (float)img.cols / (float)inWidth;
    float scale_h = (float)img.rows / (float)inHeight;

    dpuSetInputImage2(task, NODE_INPUT, img);//dpu utils function

    dpuRunTask(task); //run dpu task

    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, NODE_CONV);//get NODE_CONV:"pixl_conv" output
    int tensorSize = dpuGetTensorSize(conv_out_tensor);//96*160*2

    DPUTensor *conv_out_tensor_2 = dpuGetOutputTensor(task, NODE_OUTPUT);//get NODE_OUTPUT:"bb_out" output
    int tensorSize_2 = dpuGetTensorSize(conv_out_tensor_2);//96*160*4
    int outHeight_2 = dpuGetTensorHeight(conv_out_tensor_2);
    int outWidth_2 = dpuGetTensorWidth(conv_out_tensor_2);

    vector<float> pixel(tensorSize);//96*160*2
    vector<float> conf(tensorSize);//96*160*2
    vector<float> bb(tensorSize_2);//96*160*4

    //output data format convert
    dpuGetOutputTensorInHWCFP32(task, NODE_CONV, pixel.data(), tensorSize);
    dpuGetOutputTensorInHWCFP32(task, NODE_OUTPUT, bb.data(), tensorSize_2);

    //2-classes softmax
    softmax_2(pixel, conf);

    // get original face boxes
    vector<vector<float> > boxes;
    for (int i = 0; i < outHeight_2; i++) {
        for (int j = 0; j < outWidth_2; j++) {
            int position = i * outWidth_2 + j;
            vector<float> box;
            if (conf[position * 2 + 1] > 0.55) {
                box.push_back(bb[position * 4 + 0] + j * 4);
                box.push_back(bb[position * 4 + 1] + i * 4);
                box.push_back(bb[position * 4 + 2] + j * 4);
                box.push_back(bb[position * 4 + 3] + i * 4);
                box.push_back(conf[position * 2 + 1]);
                boxes.push_back(box);
            }
        }
    }

    // Discard overlapping boxes using NMS
    vector<vector<float> > res = NMS(boxes, 0.35);//res output

    // put detected face boxes to image
    for (size_t i = 0; i < res.size(); ++i) {
        float xmin = std::max(res[i][0] * scale_w, 0.0f);
        float ymin = std::max(res[i][1] * scale_h, 0.0f);
        float xmax = std::min(res[i][2] * scale_w, (float)img.cols);
        float ymax = std::min(res[i][3] * scale_h, (float)img.rows);
        if(((xmax - xmin > 96) && (xmax - xmin < 128)) || ((ymax - ymin > 112) && (ymax - ymin < 144)))
        {
            vector<float> coordinate;
            coordinate.push_back(xmin);
            coordinate.push_back(ymin);
            coordinate.push_back(xmax);
            coordinate.push_back(ymax);
            Facept.push_back(coordinate);
            Rect pt(xmin,ymin,xmax-xmin,ymax-ymin);
            FaceROI.push_back(img(pt));
        }
    }
}

float InnerProduct(float* vectorA, float* vectorB, int size){
    float temp = 0.0;
    for (int i = 0; i < size; i++){
        temp = temp + (*(vectorA+i)) * (*(vectorB+i));
    }
    return temp;
}

pair<int, float> WorkThread:: CPUCalcVectorAngle(float facelist[2][512], float *result) {

    float a[2];
    float inner;
    float angle; // angle is the similarity between vecA and vecB
    float q = sqrt(InnerProduct(result,result,512));
    // Calculate the inner product 
    for (int i = 0; i < 2; i++)
    {
        inner = sqrt(InnerProduct(facelist[i],facelist[i],512));
        a[i] = InnerProduct(facelist[i], result, 512)/(inner*q);
    }
    int max_idx = max_element(a,a+2) - a;
    angle = a[max_idx];
    if(angle > 0.4 && max_idx != face_index)
    {
        emit recognize_bool(max_idx);
        face_index = max_idx;
    }
    pair<int, double>p(max_idx, angle);
    return p;
}

void WorkThread:: DrawImage(Mat &frame, vector<float> face, float facelist[2][512], float* result)
{
    pair<int, double>p = CPUCalcVectorAngle(facelist, result);
    string name_str = ReadText("./name.txt", p.first);
    if(p.second > 0.4){
        rectangle(frame,Point(face[0], face[1]),Point(face[2], face[3]),Scalar(0,255,0),2,8,0);
        putText(frame, name_str,Point(face[2], face[3]),FONT_HERSHEY_PLAIN,2,Scalar(0,0,255),2,8,0);//Scalar :BGR
    }
    else{
        rectangle(frame,Point(face[0], face[1]),Point(face[2], face[3]),Scalar(0,0,255),2,8,0);
    }
}


/* get current person name
 * from name.txt
 * filename:file path
 * line:current index of name in file
 */
string WorkThread:: ReadText(string filename, int line)
{
    ifstream fin;
    fin.open(filename, ios::in);
    string strVec[11];     //文本中总共有10行
    int i = 0;
    while (!fin.eof())
    {
        string inbuf;
        getline(fin, inbuf, '\n');
        strVec[i] = inbuf;
        i = i + 1;
    }
    fin.close();
    return strVec[line];
}


/**
 * @brief Run DPU Task for ResNet50
 *
 * @param taskResnet50 - pointer to ResNet50 Task
 *
 * @return none
 */
void WorkThread:: runResnet50(DPUTask *taskResnet50,Mat &frame,vector<Mat> FaceROI, vector<vector<float>> Facept)
{
    assert(taskResnet50);
    //vector<string> kinds;

    /* Load all kinds words.*/
    float facelist[2][512] ;
    LoadVector(*facelist);  //load .txt

    /* Get channel count of the output Tensor for ResNet50 Task  */
    int channel = dpuGetOutputTensorChannel(taskResnet50, OUTPUT_NODE);
    float *FCResult = new float[channel];
    float mean_pt[3] = {127,127,127};
    for(size_t i = 0; i < Facept.size(); i++)
    {
        dpuSetInputImageWithScale(taskResnet50, INPUT_NODE, FaceROI[i], mean_pt, 0.0078125);
        dpuRunTask(taskResnet50);
        dpuGetOutputTensorInHWCFP32(taskResnet50, OUTPUT_NODE, FCResult, channel);
        DrawImage(frame,Facept[i],facelist,FCResult);
    }
    delete[] FCResult;
}
