/**
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