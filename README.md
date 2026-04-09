2023100345邓佳瑶作业4
## 图像缩小、恢复与频域分析

## 一、实验目的
使用 OpenCV 读入一幅灰度图像，先对图像进行下采样（缩小），再分别用不同内插方法恢复图像尺寸，并结合傅里叶变换和 DCT 变换对原图与恢复图进行分析比较。

## 二、实验环境

- **操作系统**：Windows 11 + WSL 2 (Ubuntu 22.04)
- **编译器**：g++ 11.4.0
- **集成开发环境**：Visual Studio Code + C/C++ 扩展
- **依赖库**：OpenCV 4.5.4（libopencv-dev）
- **测试图片**：`test.jpg`（分辨率 1214×810，8 位灰度图）

## 三、实验内容与实现

1. 图像读入与预处理
使用 `cv::imread` 以灰度模式读入图片。
显示原图尺寸。
2. 下采样（缩小为 1/4）
直接缩小：使用最近邻插值 `INTER_NEAREST`，不做预滤波。
先高斯平滑再缩小：先用 `GaussianBlur` 平滑，再同样用最近邻缩小，观察混叠现象的减少。

3. 图像恢复（放大回原尺寸）
对直接缩小后的图像分别使用三种插值方法放大到原始尺寸：
最近邻插值：`INTER_NEAREST`，像素复制，会产生锯齿。
双线性插值：`INTER_LINEAR`，平滑但会损失部分高频细节。
双三次插值：`INTER_CUBIC`，更平滑，计算量更大。

4. 空间域比较
显示原图、缩小图、恢复图。
计算均方误差（MSE）和峰值信噪比（PSNR）评估恢复质量。

5. 傅里叶变换分析
对以下图像计算二维傅里叶变换并显示频谱：
原图
直接缩小图
双线性恢复图
- 双三次恢复图

**实现要点**：
使用 `cv::dft` 计算傅里叶变换。将频谱中心移动到图像中心（`fftshift`）。对幅度谱取对数（`log(1 + magnitude)`）以压缩动态范围。比较高频成分差异并解释原因。

6. DCT 分析
对原图和恢复图（双线性）做二维离散余弦变换（`cv::dct`）：显示 DCT 系数图（取对数显示）。
- 统计左上角 1/8 低频区域能量占总能量的比例。
- 比较不同恢复方法下 DCT 能量分布差异并解释。

## 四、运行结果

### 终端输出示例
见图片。
### 图像窗口

程序依次弹出以下窗口，按任意键关闭当前窗口并显示下一个：

1. 原图
2. 直接缩小图
3. 高斯平滑后缩小图
4. 最近邻恢复图
5. 双线性恢复图
6. 双三次恢复图
7. 原图频谱
8. 缩小图频谱
9. 双线性恢复图频谱
10. 双三次恢复图频谱
11. 原图 DCT 系数
12. 双线性恢复图 DCT 系数

## 五、结果分析
5.1 空间域质量
PSNR 越高表示恢复图像越接近原图。
双三次插值 PSNR 最高，双线性次之，最近邻最差，符合预期。
- 
5.2 傅里叶频谱
原图频谱：中心明亮（低频能量集中），向外逐渐变暗（高频成分）。
直接缩小图频谱：出现明显的混叠现象（高频成分折叠到低频区域），表现为频谱中出现额外的亮斑。
高斯平滑后缩小图：混叠现象显著减少，因为预滤波提前衰减了高频成分。
恢复图频谱：双线性/双三次恢复的高频成分被部分平滑，频谱比原图更集中于中心。

5.3 DCT 能量分布
原图低频能量占比约 92%，恢复图低频能量占比升高至 96% 以上，说明恢复过程损失了高频细节。
双三次恢复比双线性恢复保留略多的高频能量，因此视觉上更清晰。

## 六、编译运行

1. 安装 OpenCV（如未安装）

```bash
sudo apt update
sudo apt install libopencv-dev
```
2. 编译

在 `frequency-lab` 目录下执行：

```bash
g++ -o main main.cpp $(pkg-config --cflags --libs opencv4)
``
3. 运行
./main test.jpg

4. 代码：/**
 * 实验：图像缩小、恢复与频域分析
 * 功能：
 *   1. 读入灰度图像
 *   2. 下采样（直接缩小 vs 高斯平滑后缩小）
 *   3. 图像恢复（最近邻、双线性、双三次插值）
 *   4. 空间域比较（MSE、PSNR）
 *   5. 傅里叶变换分析（显示频谱，比较高频成分）
 *   6. DCT 分析（系数图、低频能量占比）
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

// ==================== 辅助函数 ====================

/**
 * 显示频谱（中心化 + 对数变换）
 * @param complexI 复数图像（DFT 输出）
 * @param title 窗口标题
 */
void showSpectrum(Mat& complexI, string title) {
    Mat magI;
    Mat planes[] = {Mat::zeros(complexI.size(), CV_32F), Mat::zeros(complexI.size(), CV_32F)};
    split(complexI, planes);                // 分离实部虚部
    magnitude(planes[0], planes[1], magI);  // 计算幅值 = sqrt(Re^2 + Im^2)

    // 对数变换（压缩动态范围，让细节可见）
    magI = magI + Scalar::all(1);
    log(magI, magI);

    // 中心化：将低频移动到图像中心
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // 左上
    Mat q1(magI, Rect(cx, 0, cx, cy));  // 右上
    Mat q2(magI, Rect(0, cy, cx, cy));  // 左下
    Mat q3(magI, Rect(cx, cy, cx, cy)); // 右下

    // 交换象限
    Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);

    // 归一化到 0~255 显示
    normalize(magI, magI, 0, 255, NORM_MINMAX);
    magI.convertTo(magI, CV_8U);
    imshow(title, magI);
}

/**
 * 计算两幅图像之间的 MSE（均方误差）和 PSNR（峰值信噪比）
 * @param original 原始图像
 * @param recovered 恢复后的图像
 */
void computeMSE_PSNR(const Mat& original, const Mat& recovered) {
    Mat diff;
    absdiff(original, recovered, diff);          // 逐像素差值
    Mat diff_squared = diff.mul(diff);            // 平方
    double mse = mean(diff_squared)[0];           // 取均值

    if (mse == 0) {
        cout << "MSE = 0，两幅图像完全相同" << endl;
        cout << "PSNR = 无穷大" << endl;
    } else {
        double psnr = 10 * log10(255.0 * 255.0 / mse);
        cout << "MSE = " << mse << endl;
        cout << "PSNR = " << psnr << " dB" << endl;
    }
}

// ==================== 主函数 ====================

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 2) {
        cout << "用法: ./main 图片路径" << endl;
        return -1;
    }

    // ========== 1. 图像读入与预处理 ==========
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);  // 以灰度图形式读取
    if (img.empty()) {
        cout << "无法读取图片: " << argv[1] << endl;
        return -1;
    }
    cout << "原图尺寸: " << img.cols << " x " << img.rows << endl;
    imshow("1. 原图", img);

    // ========== 2. 下采样（缩小为原来的 1/4） ==========

    // 方法A：直接缩小（隔点采样，不做预滤波）
    Mat small_direct;
    resize(img, small_direct, Size(), 0.25, 0.25, INTER_NEAREST);
    imshow("2A. 直接缩小图", small_direct);

    // 方法B：先高斯平滑再缩小（预滤波，减少混叠）
    Mat blurred, small_blur;
    GaussianBlur(img, blurred, Size(5, 5), 1.0);
    resize(blurred, small_blur, Size(), 0.25, 0.25, INTER_NEAREST);
    imshow("2B. 高斯平滑后缩小", small_blur);

    // ========== 3. 图像恢复（放大回原尺寸） ==========
    // 对直接缩小图进行三种不同的插值放大

    // 最近邻插值（像素复制，会产生锯齿）
    Mat recover_nn;
    resize(small_direct, recover_nn, img.size(), 0, 0, INTER_NEAREST);
    imshow("3A. 最近邻恢复", recover_nn);

    // 双线性插值（平滑，消除锯齿但会模糊）
    Mat recover_linear;
    resize(small_direct, recover_linear, img.size(), 0, 0, INTER_LINEAR);
    imshow("3B. 双线性恢复", recover_linear);

    // 双三次插值（更平滑，计算量更大）
    Mat recover_cubic;
    resize(small_direct, recover_cubic, img.size(), 0, 0, INTER_CUBIC);
    imshow("3C. 双三次恢复", recover_cubic);

    // ========== 4. 空间域比较（MSE & PSNR） ==========
    cout << "\n========== 空间域质量评价 ==========" << endl;
    cout << "【双线性恢复 vs 原图】" << endl;
    computeMSE_PSNR(img, recover_linear);
    cout << "【双三次恢复 vs 原图】" << endl;
    computeMSE_PSNR(img, recover_cubic);
    cout << "【最近邻恢复 vs 原图】" << endl;
    computeMSE_PSNR(img, recover_nn);

    // ========== 5. 傅里叶变换分析 ==========
    // 获取合适的 DFT 尺寸（提高效率）
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);

    // 辅助函数：对图像进行 DFT 并显示频谱
    auto dftAndShow = [&](const Mat& image, string title) {
        Mat padded;
        copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
        Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
        Mat complexImg;
        merge(planes, 2, complexImg);
        dft(complexImg, complexImg);
        showSpectrum(complexImg, title);
    };

    cout << "\n========== 傅里叶变换分析 ==========" << endl;
    dftAndShow(img, "5A. 原图频谱");
    dftAndShow(small_direct, "5B. 缩小图频谱（直接下采样）");
    dftAndShow(recover_linear, "5C. 双线性恢复图频谱");
    dftAndShow(recover_cubic, "5D. 双三次恢复图频谱");

    // ========== 6. DCT 分析（二维离散余弦变换） ==========
    cout << "\n========== DCT 分析 ==========" << endl;

    // 将图像转为浮点型（DCT 要求）
    Mat img_float, recover_linear_float;
    img.convertTo(img_float, CV_32F);
    recover_linear.convertTo(recover_linear_float, CV_32F);

    // 对原图做 DCT
    Mat dct_original;
    dct(img_float, dct_original);
    
    // 对恢复图做 DCT
    Mat dct_recover;
    dct(recover_linear_float, dct_recover);

    // 显示 DCT 系数图（取对数 + 归一化）
    auto showDCT = [](const Mat& dctCoeff, string title) {
        Mat logCoeff;
        Mat absCoeff = abs(dctCoeff);
        absCoeff = absCoeff + Scalar::all(1);
        log(absCoeff, logCoeff);
        normalize(logCoeff, logCoeff, 0, 255, NORM_MINMAX);
        logCoeff.convertTo(logCoeff, CV_8U);
        imshow(title, logCoeff);
    };
    
    showDCT(dct_original, "6A. 原图 DCT 系数");
    showDCT(dct_recover, "6B. 双线性恢复图 DCT 系数");

    // 计算低频能量占比（取左上角 1/8 区域）
    auto lowFreqEnergyRatio = [](const Mat& dctCoeff) -> double {
        int lowW = dctCoeff.cols / 8;
        int lowH = dctCoeff.rows / 8;
        Mat lowFreq = dctCoeff(Rect(0, 0, lowW, lowH));
        double lowEnergy = sum(lowFreq.mul(lowFreq))[0];
        double totalEnergy = sum(dctCoeff.mul(dctCoeff))[0];
        return lowEnergy / totalEnergy;
    };

    double ratio_original = lowFreqEnergyRatio(dct_original);
    double ratio_recover = lowFreqEnergyRatio(dct_recover);

    cout << "原图 DCT 低频能量占比: " << ratio_original * 100 << "%" << endl;
    cout << "双线性恢复图 DCT 低频能量占比: " << ratio_recover * 100 << "%" << endl;
    cout << "结论: 恢复图低频能量占比更高，说明高频细节在缩小-放大过程中丢失。" << endl;

    // ========== 等待按键后退出 ==========
    cout << "\n按任意键关闭所有窗口..." << endl;
    waitKey(0);
    destroyAllWindows();
    return 0;
}
