#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
	
    ui->setupUi(this);

    //define a object to detect temper
    tem = new TestTem();       //检测温度对象

    //start connect to severe teminal
    clint = new Tcp_to_seve(this);
    clint->start();
    connect(clint, &Tcp_to_seve::show_file_open_error, this, &MainWindow::show_file_error_message);

    //set ui backimage
    iconImage = new QImage();              //set background image
    iconImage->load(":/new/prefix1/images/icon.jpg");
    ui->iconLabel->setPixmap(QPixmap::fromImage(*iconImage));

    backImage = new QImage();
    backImage->load(":/new/prefix1/images/backimage.jpg");
    ui->backgroundLabel->setPixmap(QPixmap::fromImage(*backImage));

    //ui->backLabel1->setStyleSheet("QLabel{background-color:rgb(0,0,255);}");


//    QPalette pal(this->palette());
//    pal.setColor(QPalette::Background, Qt::white); //设置背景黑色
//    this->setAutoFillBackground(true);
//    this->setPalette(pal);

//    QPalette pe1;
//    pe1.setColor(QPalette::WindowText,Qt::white);
//    ui->titleLabel->setPalette(pe1);
    ui->titleLabel->setStyleSheet("color:rgb(0,183,191);");


    //ui->describeText->setStyleSheet("background-color:white;");
    //ui->describeText->setStyleSheet("QTextBrowser{border-width:0;border-style:outset}");
    QPalette pe2;
    pe2.setColor(QPalette::WindowText,Qt::white);
    ui->describeText->setPalette(pe2);

    //timer control
    showTime = new QTimer(this);
    showTime->start(1000);

    getImageTimer = new QTimer(this);
    getImageTimer->start(50);

    showImageTimer = new QTimer(this);
    showImageTimer->start(50);

    //open camera
    this->openCamara();

    //get face images
    connect(getImageTimer, &QTimer::timeout, this, &MainWindow::inputImage);

    //start to face detection and process face images
    this->main_face_detection();

    //show images which has been processed on gui
    connect(showImageTimer, &QTimer::timeout, this, &MainWindow::showImage);

    //quit thread after close window
    connect(this, &MainWindow::destroyed, this, &MainWindow::stopThread);   //quit thread after close window

    //show current time on gui
    connect(showTime, &QTimer::timeout, this, &MainWindow::showCurrentTime);

}


MainWindow::~MainWindow()
{
    delete ui;
}


/* detect temperature and save datas
 * after recognize face.
 * parameter:
 * nameId:store index of name of current person
 */
void MainWindow::coltrol_tcp_tem(int nameId)
{
    double result_tem;
    string current_name;
    QString qt_time_str = QDateTime::currentDateTime().toString("yyyy.MM.dd hh:mm:ss");
    string current_time = qt_time_str.toStdString();
    ofstream os;

    while(1)
    {
        //detect tem
        //result_tem = tem->culTem();
        result_tem = 14.9;
        if(result_tem >= 12.0 && result_tem <= 17.2)
        {
            ui->temLabel->setStyleSheet("QLabel{color:rgb(0,183,191);}");
            ui->temLabel->setText("温度正常");
            current_name = ReadText("./name.txt", nameId);
            os.open("./related_file/personDatas.txt");  //
	    std::string str = to_string(nameId);
	    std::string newstr = std::string(3-str.length(),'0') + str;
            QString qstr = QString::fromStdString(newstr);
            clint->sock->write(qstr.toUtf8());
            string str_file = current_name + "[" + to_string(nameId) + "] " + current_time + " " + "normal";
            os << str_file;
            os.close();
            break;
        }

        else
        {
            ui->temLabel->setStyleSheet("QLabel{color:rgb(200,0,0);}");
            ui->temLabel->setText("温度异常");
            continue;
        }
    }

}


/* get current person name
 * from name.txt
 * filename:file path
 * line:current index of name in file
 */
string MainWindow:: ReadText(string filename, int line)
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
    return strVec[line];
}


//show error message after failed to open file
void MainWindow::show_file_error_message()
{
    QMessageBox::information(this, "提示","只读方式打开文件失败");
}


//show current time on gui
void MainWindow::showCurrentTime()         //show current time and date
{
//    QPalette pe3;
//    pe3.setColor(QPalette::WindowText,Qt::blue);
    ui->timeLabel->setText(QDateTime::currentDateTime().toString("hh : mm"));
//    ui->timeLabel->setPalette(pe3);
    ui->timeLabel->setStyleSheet("color:rgb(0,183,191);");


//    QPalette pe1;
//    pe1.setColor(QPalette::WindowText,Qt::white);
    ui->dateLabel->setText(QDateTime::currentDateTime().toString("yyyy/MM/dd"));
//    ui->dateLabel->setPalette(pe1);
    ui->dateLabel->setStyleSheet("color:rgb(0,183,191);");
}


/* convert Mat image to QImage
 * src:mat image
 */
QImage MainWindow::cvMat2QImage(const Mat &src)  //mat图像转换成QImage
{

        //CV_8UC1 8位无符号的单通道---灰度图片
        if(src.type() == CV_8UC1)
        {
            //使用给定的大小和格式构造图像
            //QImage(int width, int height, Format format)
            QImage qImage(src.cols,src.rows,QImage::Format_Indexed8);
            //扩展颜色表的颜色数目
            qImage.setColorCount(256);

            //在给定的索引设置颜色
            for(int i = 0; i < 256; i ++)
            {
                //得到一个黑白图
                qImage.setColor(i,qRgb(i,i,i));
            }
            //复制输入图像,data数据段的首地址
            uchar *pSrc = src.data;
            //
            for(int row = 0; row < src.rows; row ++)
            {
                //遍历像素指针
                uchar *pDest = qImage.scanLine(row);
                //从源src所指的内clint->start();

                //字节到目标dest所指的内存地址的起始位置中
                memcmp(pDest,pSrc,src.cols);
                //图像层像素地址
                pSrc += src.step;
            }
            return qImage;
        }
        //为3通道的彩色图片
        else if(src.type() == CV_8UC3)
        {
            //得到图像的的首地址
            const uchar *pSrc = (const uchar*)src.data;
            //以src构造图片
            QImage qImage(pSrc,src.cols,src.rows,src.step,QImage::Format_RGB888);
            //在不改变实际图像数据的条件下，交换红蓝通道
            return qImage.rgbSwapped();
        }
        //四通道图片，带Alpha通道的RGB彩色图像
        else if(src.type() == CV_8UC4)
        {
            const uchar *pSrc = (const uchar*)src.data;
            QImage qImage(pSrc, src.cols, src.rows, src.step, QImage::Format_ARGB32);
            //返回图像的子区域作为一个新图像
            return qImage.copy();
        }

        else
        {
            return QImage();
        }

}


//open camera
void MainWindow::openCamara()
{
    if(cap.isOpened())
    {
        cap.release();
    }
    cap.open(0, CAP_V4L);

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);
    cap.set(CV_CAP_PROP_FPS, 20);

}


//get face images from camera
void MainWindow::inputImage()
{
    // image index of input video
    Mat img;
    cap >> img;
    if (img.empty()) {
        cerr << "Fail to read image from cap!" << endl;
        cap.release();
        exit(0);
    }
    mtxQueueInput.lock();
    queueInput.push(make_pair(idxInputImage++, img));
    if (queueInput.size() >= 100) {
        mtxQueueInput.unlock();
        cout << "[Warning]input queue size is " << queueInput.size() << endl;
        // Sleep for a moment
        usleep(200);
    } else {
         mtxQueueInput.unlock();
    }
}


//detect face and process image
void MainWindow::main_face_detection()
{
	cout << "begin main_face_detection" << endl;
    // Attach to DPU driver and prepare for running
    dpuOpen();

    // Doing face detection.
    //FDR();

    t1 = new WorkThread();
    t2 = new WorkThread();

    connect(t1, &WorkThread::recognize_bool, this, &MainWindow::coltrol_tcp_tem);
    //connect(t2, &WorkThread::recognize_bool, this, &MainWindow::coltrol_tcp_tem);

    t1->start();
    t2->start();

	//cout << "end main_face_detection" << endl;
}


//show images on gui
void MainWindow::showImage()
{
    mtxQueueShow.lock();
    if (queueShow.empty())
    {  // no image in display queue
        mtxQueueShow.unlock();
        usleep(10000);  // Sleep for a moment
        return;
    }
    else if (idxShowImage.load() == queueShow.top().first)
    {
        //cv::imshow("Face Detection @Xilinx DPU", queueShow.top().second);  // Display image
        imag = cvMat2QImage(queueShow.top().second);
        ui->camera->setPixmap(QPixmap::fromImage(imag));
        idxShowImage++;
        queueShow.pop();
        mtxQueueShow.unlock();
    }
    else{
        mtxQueueShow.unlock();
    }
}



void MainWindow::stopThread()   //deal thread after close window
{

//    for(auto i = 0; i < workerNum; i++)
//    {
//        workers[i]->quit();
//    }


//    dpuDestroyTask(detectfaceTask);
//    dpuDestroyTask(recogfaceTask);

//    cout << "1" << endl;
//        // Destroy DPU Kernel & free resources
//    dpuDestroyKernel(DenseBoxKernel);

//    cout << "2" << endl;
//    dpuDestroyKernel(ResNet50Kernel);

        // Dettach from DPU driver & release resources
    // //cout << "3" << endl;
    dpuClose();
}

















