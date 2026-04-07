#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {

    cout << "🚀 Starting...\n";

    // ----------------------------
    // LOAD MODEL
    // ----------------------------
    Net net = readNetFromONNX("model.onnx");

    if (net.empty()) {
        cout << "❌ Model load failed\n";
        return -1;
    }

    cout << "✅ Model loaded successfully\n";

    // ----------------------------
    // LOAD IMAGE
    // ----------------------------
    Mat img = imread("test.jpg");

    if (img.empty()) {
        cout << "❌ Image not found\n";
        return -1;
    }

    cout << "✅ Image loaded\n";

    resize(img, img, Size(256, 256));

    // ----------------------------
    // PREPROCESS
    // ----------------------------
    Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255.0);

    Mat blob = blobFromImage(img_float, 1.0, Size(256,256), Scalar(), true, false);

    net.setInput(blob);

    cout << "⚡ Running inference...\n";

    // ----------------------------
    // INFERENCE
    // ----------------------------
    Mat output = net.forward();

    cout << "✅ Inference done\n";

    // ----------------------------
    // OUTPUT PROCESSING
    // ----------------------------
    Mat mask(256, 256, CV_32F, output.ptr<float>());

    // Debug values
    double minVal, maxVal;
    minMaxLoc(mask, &minVal, &maxVal);

    cout << "📊 Min value: " << minVal << endl;
    cout << "📊 Max value: " << maxVal << endl;

    // 🔥 Save raw mask (for debugging)
    Mat raw_mask;
    mask.convertTo(raw_mask, CV_8U, 255);
    imwrite("raw_mask.png", raw_mask);
    cout << "🧪 Raw mask saved as raw_mask.png\n";

    // ✅ FINAL THRESHOLD (LOW CONFIDENCE MODEL)
    Mat binary;
    threshold(mask, binary, 0.01, 1.0, THRESH_BINARY);

    binary.convertTo(binary, CV_8U, 255);

    imwrite("mask.png", binary);
    cout << "🖼️ Mask saved as mask.png\n";

    // ----------------------------
    // OVERLAY
    // ----------------------------
    Mat color_mask;
    applyColorMap(binary, color_mask, COLORMAP_JET);

    Mat img_uint8;
    img.convertTo(img_uint8, CV_8U);

    Mat overlay;
    addWeighted(img_uint8, 0.7, color_mask, 0.3, 0, overlay);

    imwrite("overlay.png", overlay);
    cout << "🎨 Overlay saved as overlay.png\n";

    cout << "✅ DONE\n";

    return 0;
}