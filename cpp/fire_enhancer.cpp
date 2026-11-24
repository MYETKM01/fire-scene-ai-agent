#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <windows.h>
#include <omp.h>
#include <iomanip>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace cv;
using namespace std;

#define CLAMP(x) ((x) < 0 ? 0 : ((x) > 255 ? 255 : (x)))

// =========================================================================
//  PART A: 基礎輔助函式
// =========================================================================
void fast_rgb2gray_hw(const Mat& src, Mat& dst) {
    if (dst.empty() || dst.size() != src.size() || dst.type() != CV_8UC1)
        dst.create(src.size(), CV_8UC1);
    int rows = src.rows, cols = src.cols;
    if (src.isContinuous() && dst.isContinuous()) { cols *= rows; rows = 1; }
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        const uchar* pSrc = src.ptr<uchar>(i);
        uchar* pDst = dst.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            int b = pSrc[3*j], g = pSrc[3*j+1], r = pSrc[3*j+2];
            pDst[j] = (uchar)((77*r + 150*g + 29*b) >> 8);
        }
    }
}

inline void rgb2yuv_fast(unsigned char r, unsigned char g, unsigned char b, int* y, int* u, int* v) {
    *y = (( 66*r + 129*g +  25*b + 128) >> 8) + 16;
    *u = ((-38*r -  74*g + 112*b + 128) >> 8) + 128;
    *v = ((112*r -  94*g -  18*b + 128) >> 8) + 128;
}

inline void yuv2rgb_fast(int y, int u, int v, unsigned char* r, unsigned char* g, unsigned char* b) {
    int c = y-16, d = u-128, e = v-128;
    *r = (unsigned char)CLAMP((298*c         + 409*e + 128) >> 8);
    *g = (unsigned char)CLAMP((298*c - 100*d - 208*e + 128) >> 8);
    *b = (unsigned char)CLAMP((298*c + 516*d         + 128) >> 8);
}

// =========================================================================
//  PART B: 三類 Mask 生成 (沿用 18 版本，保留優秀的草地/天空濾除)
// =========================================================================
void get_triclass_mask_temporal(const Mat& src, Mat& fire_mask, Mat& smoke_mask, Mat& prev_smoke_mask, float temporal_alpha = 0.72f) {
    int h = src.rows, w = src.cols;
    static Mat hsv, gray, color_mask; 

    cvtColor(src, hsv, COLOR_BGR2HSV);
    Mat mask1, mask2;
    inRange(hsv, Scalar(0, 40, 100), Scalar(45, 255, 255), mask1);
    inRange(hsv, Scalar(150, 40, 100), Scalar(180, 255, 255), mask2);
    color_mask = mask1 | mask2;

    fast_rgb2gray_hw(src, gray);
    Mat binary;
    threshold(gray, binary, 190, 255, THRESH_BINARY);

    Mat mask_result;
    bitwise_and(binary, color_mask, mask_result);

    Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));
    dilate(mask_result, mask_result, kernel5);
    dilate(mask_result, mask_result, kernel5);
    erode(mask_result, mask_result, kernel5);
    erode(mask_result, mask_result, kernel5);

    {
        vector<vector<Point>> contours;
        findContours(mask_result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        fire_mask = Mat::zeros(mask_result.size(), CV_8UC1);
        for (const auto& cnt : contours) {
            if (cv::contourArea(cnt) > 80) drawContours(fire_mask, vector<vector<Point>>{cnt}, -1, 255, -1);
        }
    }
    blur(fire_mask, fire_mask, Size(3,3));

    Mat smoke_raw = Mat::zeros(h, w, CV_8UC1);
    
    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        const Vec3b* hsv_ptr = hsv.ptr<Vec3b>(y);
        const uchar* gray_ptr = gray.ptr<uchar>(y);
        const uchar* src_ptr  = src.ptr<uchar>(y);
        uchar* smoke_ptr = smoke_raw.ptr<uchar>(y);

        for (int x = 0; x < w; ++x) {
            int S = hsv_ptr[x][1];
            int Y = gray_ptr[x];
            int b = src_ptr[3*x], g = src_ptr[3*x+1], r = src_ptr[3*x+2];
            int maxc = max({r, g, b}), minc = min({r, g, b});
            int diff = maxc - minc;
            
            bool is_strict_grey = (diff < 15);
            bool is_bright = (Y > 120);
            bool is_low_sat = (S < 30);
            
            bool is_sky = (b > r + 10);
            bool is_veg = (g > r + 5) && (g > b + 5);
            bool is_fire = fire_mask.at<uchar>(y, x) > 128;
            
            if (is_strict_grey && is_bright && is_low_sat && !is_sky && !is_veg && !is_fire) {
                smoke_ptr[x] = 255;
            }
        }
    }

    Mat kernel_huge = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
    morphologyEx(smoke_raw, smoke_raw, MORPH_OPEN, kernel_huge);
    morphologyEx(smoke_raw, smoke_raw, MORPH_CLOSE, kernel_huge);

    {
        vector<vector<Point>> contours_s;
        findContours(smoke_raw, contours_s, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        smoke_raw = Mat::zeros(h, w, CV_8UC1);
        for (const auto& cnt : contours_s)
            if (cv::contourArea(cnt) > 1000) drawContours(smoke_raw, vector<vector<Point>>{cnt}, -1, 255, -1);
    }

    if (prev_smoke_mask.empty()) {
        smoke_raw.copyTo(prev_smoke_mask);
        smoke_raw.copyTo(smoke_mask);
    } else {
        Mat smoothed;
        addWeighted(smoke_raw, temporal_alpha, prev_smoke_mask, 1.0f - temporal_alpha, 0.0, smoothed);
        threshold(smoothed, smoke_mask, 80, 255, THRESH_BINARY);
        smoke_mask.copyTo(prev_smoke_mask);
    }
}

// =========================================================================
//  PART C: DCP 去霧 
// =========================================================================
void compute_dark_channel(const Mat& src_bgr, Mat& dark_ch, int patch_size) {
    int rows = src_bgr.rows, cols = src_bgr.cols;
    dark_ch.create(rows, cols, CV_8UC1);
    Mat min_bgr(rows, cols, CV_8UC1);
    #pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        const uchar* p = src_bgr.ptr<uchar>(y);
        uchar* q = min_bgr.ptr<uchar>(y);
        for (int x = 0; x < cols; x++) q[x] = min({p[3*x], p[3*x+1], p[3*x+2]});
    }
    Mat kernel = getStructuringElement(MORPH_RECT, Size(patch_size, patch_size));
    erode(min_bgr, dark_ch, kernel);
}

void estimate_atmospheric_light(const Mat& src_bgr, const Mat& dark_ch, const uchar* fire_flag, Vec3f& A) {
    int total = src_bgr.rows * src_bgr.cols;
    vector<pair<int,int>> dc_vals; dc_vals.reserve(total);
    const uchar* dc_ptr = dark_ch.data;
    for (int i = 0; i < total; i++) {
        if (fire_flag[i] == 0) dc_vals.push_back({dc_ptr[i], i});
    }
    if (dc_vals.empty()) { A = Vec3f(200, 200, 200); return; }

    int top_n = max(1, (int)dc_vals.size() / 1000);  
    partial_sort(dc_vals.begin(), dc_vals.begin() + top_n, dc_vals.end(),
                 [](const pair<int,int>& a, const pair<int,int>& b){ return a.first > b.first; });

    Vec3d sum(0, 0, 0); const uchar* src_ptr = src_bgr.data;
    for (int k = 0; k < top_n; k++) {
        int idx = dc_vals[k].second;
        sum[0] += src_ptr[idx*3]; sum[1] += src_ptr[idx*3+1]; sum[2] += src_ptr[idx*3+2];
    }
    A[0] = (float)(sum[0] / top_n); A[1] = (float)(sum[1] / top_n); A[2] = (float)(sum[2] / top_n);
}

void apply_dark_channel_dehaze_fast(const Mat& src_mat, unsigned char* dst_data, const uchar* smoke_alpha, const uchar* fire_flag) {
    int w = src_mat.cols, h = src_mat.rows;
    int scale = 4; 
    Mat src_small; resize(src_mat, src_small, Size(w/scale, h/scale), 0, 0, INTER_AREA);
    
    Mat fire_mask_small;
    resize(Mat(h, w, CV_8UC1, (void*)fire_flag), fire_mask_small, Size(w/scale, h/scale), 0, 0, INTER_NEAREST);
    Mat dark_small; compute_dark_channel(src_small, dark_small, 5);
    
    Vec3f A; estimate_atmospheric_light(src_small, dark_small, fire_mask_small.data, A);
    A[0] = min(A[0], 210.0f); A[1] = min(A[1], 210.0f); A[2] = min(A[2], 210.0f);

    Mat t_map_small, normalized(src_small.size(), CV_32FC3);
    #pragma omp parallel for
    for (int y = 0; y < src_small.rows; y++) {
        const uchar* p = src_small.ptr<uchar>(y); float* pn = normalized.ptr<float>(y);
        for (int x = 0; x < src_small.cols; x++) {
            pn[3*x] = p[3*x]/max(1.0f, A[0]); pn[3*x+1] = p[3*x+1]/max(1.0f, A[1]); pn[3*x+2] = p[3*x+2]/max(1.0f, A[2]);
        }
    }
    Mat norm_dc(src_small.size(), CV_32FC1);
    #pragma omp parallel for
    for (int y = 0; y < src_small.rows; y++) {
        const float* p = normalized.ptr<float>(y); float* q = norm_dc.ptr<float>(y);
        for (int x = 0; x < src_small.cols; x++) q[x] = min({p[3*x], p[3*x+1], p[3*x+2]});
    }
    Mat norm_dc_8u, eroded; norm_dc.convertTo(norm_dc_8u, CV_8UC1, 255.0f);
    erode(norm_dc_8u, eroded, getStructuringElement(MORPH_RECT, Size(5, 5)));
    t_map_small.create(src_small.size(), CV_32FC1);
    #pragma omp parallel for
    for (int y = 0; y < src_small.rows; y++) {
        const uchar* pe = eroded.ptr<uchar>(y); float* pt = t_map_small.ptr<float>(y);
        for (int x = 0; x < src_small.cols; x++) pt[x] = 1.0f - 0.90f * (pe[x] / 255.0f);
    }
    Mat gray_small, gray_f_small; cvtColor(src_small, gray_small, COLOR_BGR2GRAY);
    gray_small.convertTo(gray_f_small, CV_32FC1, 1.0f/255.0f);
    Mat mean_I, mean_p, mean_Ip, mean_II;
    boxFilter(gray_f_small, mean_I, -1, Size(17, 17)); boxFilter(t_map_small, mean_p, -1, Size(17, 17));
    boxFilter(gray_f_small.mul(t_map_small), mean_Ip, -1, Size(17, 17)); boxFilter(gray_f_small.mul(gray_f_small), mean_II, -1, Size(17, 17));
    Mat a = (mean_Ip - mean_I.mul(mean_p)) / (mean_II - mean_I.mul(mean_I) + 1e-3f);
    Mat b = mean_p - a.mul(mean_I);
    boxFilter(a, mean_I, -1, Size(17, 17)); boxFilter(b, mean_p, -1, Size(17, 17));
    Mat t_refined_small = mean_I.mul(gray_f_small) + mean_p;
    Mat t_refined; resize(t_refined_small, t_refined, Size(w, h), 0, 0, INTER_LINEAR);

    float t_min = 0.35f; const uchar* ps = src_mat.data;
    
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        const uchar* pm = smoke_alpha + y * w; 
        const float* pt = t_refined.ptr<float>(y);
        int row_off = y * w * 3;
        for (int x = 0; x < w; x++) {
            int idx = row_off + 3*x;
            if (pm[x] == 0) continue; 
            
            float t = max(pt[x], t_min);
            float inv_t = 1.0f / t;
            float alpha = pm[x] / 255.0f;
            
            for (int c = 0; c < 3; c++) {
                float dehazed = ((float)ps[idx+c] - A[c]) * inv_t + A[c];
                dst_data[idx+c] = (uchar)CLAMP(ps[idx+c] * (1.0f - alpha) + dehazed * alpha);
            }
        }
    }
}

// =========================================================================
//  PART D: CLAHE (無縫平滑，零網格)
// =========================================================================
#define TILES_X 8
#define TILES_Y 8
#define BINS    256
unsigned char tile_cdfs[TILES_Y][TILES_X][BINS];

void calculate_grid_cdfs_bidirectional(int* Y_plane, const uchar* fire_flag, int w, int h, int base_clip_limit) {
    int tile_w = w / TILES_X, tile_h = h / TILES_Y;
    #pragma omp parallel for collapse(2)
    for (int ty = 0; ty < TILES_Y; ty++) {
        for (int tx = 0; tx < TILES_X; tx++) {
            int hist[BINS] = {0};
            int start_x = tx*tile_w, end_x = (tx==TILES_X-1) ? w : start_x+tile_w;
            int start_y = ty*tile_h, end_y = (ty==TILES_Y-1) ? h : start_y+tile_h;
            
            int valid_area = 0;
            long long tile_Y_sum = 0; 

            for (int y = start_y; y < end_y; y++) {
                int row_offset = y * w;
                for (int x = start_x; x < end_x; x++) {
                    if (fire_flag[row_offset+x] == 0) {
                        int y_val = Y_plane[row_offset+x];
                        int idx = CLAMP((y_val - 16) * 255 / 219);
                        hist[idx]++; 
                        valid_area++;
                        tile_Y_sum += y_val;
                    }
                }
            }
            if (valid_area == 0) { for (int i = 0; i < BINS; i++) tile_cdfs[ty][tx][i] = i; continue; }

            int avg_Y = tile_Y_sum / valid_area;
            int dynamic_clip = base_clip_limit;
            if (avg_Y > 20) {
                dynamic_clip += (avg_Y - 20) / 8; 
            } else {
                dynamic_clip = max(1, base_clip_limit - 1);
            }
            dynamic_clip = min(dynamic_clip, 16); 

            int clipped = 0;
            for (int i = 0; i < BINS; i++) { 
                if (hist[i] > dynamic_clip) { 
                    clipped += hist[i] - dynamic_clip; 
                    hist[i] = dynamic_clip; 
                } 
            }
            int redist = clipped / BINS;
            for (int i = 0; i < BINS; i++) hist[i] += redist;
            int sum = 0; float scale = 255.0f / valid_area;
            for (int i = 0; i < BINS; i++) { sum += hist[i]; tile_cdfs[ty][tx][i] = (uchar)(sum * scale); }
        }
    }

    for(int iter=0; iter<2; iter++) {
        for (int ty = 0; ty < TILES_Y; ty++)
            for (int tx = 1; tx < TILES_X-1; tx++)
                for (int k = 0; k < BINS; k++) 
                    tile_cdfs[ty][tx][k] = (tile_cdfs[ty][tx-1][k] + (tile_cdfs[ty][tx][k] << 1) + tile_cdfs[ty][tx+1][k]) >> 2;
        for (int tx = 0; tx < TILES_X; tx++)
            for (int ty = 1; ty < TILES_Y-1; ty++)
                for (int k = 0; k < BINS; k++) 
                    tile_cdfs[ty][tx][k] = (tile_cdfs[ty-1][tx][k] + (tile_cdfs[ty][tx][k] << 1) + tile_cdfs[ty+1][tx][k]) >> 2;
    }
}

void apply_clahe_interpolation(int* Y_plane, int w, int h) {
    int tile_w = w / TILES_X, tile_h = h / TILES_Y;
    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        float ty_f = ((float)y / tile_h) - 0.5f;
        int ty1 = (int)floor(ty_f);
        float y_w = ty_f - ty1; 
        
        int ty1_c = max(0, min(TILES_Y - 1, ty1));
        int ty2_c = max(0, min(TILES_Y - 1, ty1 + 1));

        for (int x = 0; x < w; x++) {
            float tx_f = ((float)x / tile_w) - 0.5f;
            int tx1 = (int)floor(tx_f);
            float x_w = tx_f - tx1;
            
            int tx1_c = max(0, min(TILES_X - 1, tx1));
            int tx2_c = max(0, min(TILES_X - 1, tx1 + 1));

            int idx = CLAMP((Y_plane[y*w+x]-16)*255/219);
            float top = tile_cdfs[ty1_c][tx1_c][idx]*(1.0f-x_w) + tile_cdfs[ty1_c][tx2_c][idx]*x_w;
            float bot = tile_cdfs[ty2_c][tx1_c][idx]*(1.0f-x_w) + tile_cdfs[ty2_c][tx2_c][idx]*x_w;
            Y_plane[y*w+x] = CLAMP((int)(top*(1.0f-y_w) + bot*y_w)*219/255 + 16);
        }
    }
}

float current_dynamic_gamma = 0.6f;

// =========================================================================
//  PART E: 核心處理 (🔥 Final 版：保留低光，斬斷純黑網格)
// =========================================================================
void thesis_core_process(unsigned char* src, unsigned char* dst, unsigned char* out_mask, int w, int h) {
    static Mat prev_smoke_mask;
    int total = w * h;

    Mat src_mat(h, w, CV_8UC3, src);
    Mat fire_mask_mat, smoke_mask_mat; 
    get_triclass_mask_temporal(src_mat, fire_mask_mat, smoke_mask_mat, prev_smoke_mask, 0.72f);

    Mat fire_alpha, smoke_alpha;
    GaussianBlur(fire_mask_mat, fire_alpha, Size(25, 25), 0);
    GaussianBlur(smoke_mask_mat, smoke_alpha, Size(35, 35), 0);

    static std::vector<unsigned char> fire_flag_vec, smoke_flag_vec, dehazed_buf_vec;
    static std::vector<int> Y_vec, U_vec, V_vec, Y_orig_vec;

    if (fire_flag_vec.size() < (size_t)total) {
        fire_flag_vec.resize(total); smoke_flag_vec.resize(total); dehazed_buf_vec.resize(total * 3);
        Y_vec.resize(total); U_vec.resize(total); V_vec.resize(total); 
        Y_orig_vec.resize(total); 
    }

    unsigned char* fire_flag = fire_flag_vec.data();
    unsigned char* smoke_flag = smoke_flag_vec.data();
    unsigned char* dehazed_buf = dehazed_buf_vec.data();
    int* Y = Y_vec.data(); int* U = U_vec.data(); int* V = V_vec.data(); 
    int* Y_orig = Y_orig_vec.data(); 

    const uchar* fp = fire_mask_mat.data; 
    const uchar* sp = smoke_alpha.data;
    
    int smoke_pixel_count = 0;
    #pragma omp parallel for reduction(+:smoke_pixel_count)
    for (int i = 0; i < total; i++) {
        fire_flag[i] = (fp[i] > 128) ? 1 : 0;
        smoke_flag[i] = sp[i]; 
        out_mask[i] = fire_flag[i]; 
        if (sp[i] > 10) smoke_pixel_count++;
    }

    memcpy(dehazed_buf, src, total * 3);

    float smoke_ratio = (float)smoke_pixel_count / total;
    if (smoke_ratio > 0.02f) {
        apply_dark_channel_dehaze_fast(src_mat, dehazed_buf, smoke_flag, fire_flag);
    }

    long long sum_Y_non_fire = 0;
    int non_fire_count = 0;
    int min_Y_val = 255;

    #pragma omp parallel for reduction(+:sum_Y_non_fire, non_fire_count) reduction(min:min_Y_val)
    for (int i = 0; i < total; i++) {
        if (smoke_flag[i] > 0) {
            rgb2yuv_fast(dehazed_buf[i*3+2], dehazed_buf[i*3+1], dehazed_buf[i*3], &Y[i], &U[i], &V[i]);
        } else {
            rgb2yuv_fast(src[i*3+2], src[i*3+1], src[i*3], &Y[i], &U[i], &V[i]);
        }
        Y_orig[i] = Y[i]; 
        
        if (fire_flag[i] == 0) {
            sum_Y_non_fire += Y_orig[i];
            non_fire_count++;
            if (Y_orig[i] < min_Y_val) min_Y_val = Y_orig[i];
        }
    }

    int fire_count = total - non_fire_count;
    float ambient_Y = non_fire_count > 0 ? (float)sum_Y_non_fire / non_fire_count : 128.0f;
    float hazard_level = min(1.0f, (smoke_ratio * 4.0f) + ((float)fire_count / total * 10.0f));

    float ambient_idx = min(((sum_Y_non_fire/(float)max(1, non_fire_count)-16.0f)/219.0f)*1.5f + (fire_count/(float)total)*3.0f, 1.0f);
    
    current_dynamic_gamma = 0.40f + 0.50f * ambient_idx;

    // 保留 18 優秀的 shadow_boost，強化低光細節
    float pow_lut[1025]; 
    for (int i = 0; i <= 1024; i++) {
        float b_val = i / 1024.0f;
        float smooth_b = b_val * b_val * (3.0f - 2.0f * b_val); 
        float base_curve = pow(smooth_b, current_dynamic_gamma);
        float shadow_boost = 1.5f * (1.0f - ambient_idx) * b_val * exp(-6.0f * b_val);
        pow_lut[i] = min(1.0f, base_curve + shadow_boost); 
    }

    #pragma omp parallel for
    for (int i=0; i<total; i++) {
        if(fire_flag[i]) continue; 
        
        float orig = max(0.0f, min(1.0f, (Y[i]-16)/219.0f));
        float mapped_orig = pow_lut[min(1024, max(0, (int)(orig * 1024.0f)))];
        Y[i] = CLAMP(mapped_orig * 219.0f + 16.0f);
    }

    int base_clahe_clip = max(3, (int)(3.0f + 4.0f * hazard_level));
    calculate_grid_cdfs_bidirectional(Y, fire_flag, w, h, base_clahe_clip);
    apply_clahe_interpolation(Y, w, h);
    
    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        if(fire_flag[i]) Y[i] = Y_orig[i]; 
    }

    const uchar* f_alpha_ptr = fire_alpha.data;

    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        unsigned char r_enh, g_enh, b_enh;
        yuv2rgb_fast(Y[i], U[i], V[i], &r_enh, &g_enh, &b_enh);
        
        int y_val = Y_orig[i];
        
        // 1. 火源保護 (0~255)
        int f_a = f_alpha_ptr[i]; 
        
        // 2. 💡高光與原生光暈保護 (避免原生光暈被 Gamma/CLAHE 放大)
        int l_a = 0;
        if (y_val > 190) {
            l_a = 255; // 核心光源與強烈光暈 100% 保留原圖
        } else if (y_val > 100) {
            // 將保護下限從 150 放寬到 100。
            // 這段區間通常包含了光源外圍的「漸層散光」。
            // 提早進行線性保護，能完美壓制光暈向外擴張。
            l_a = ((y_val - 100) * 255) / 90; 
        }

        // 3. 純黑防護 (斬斷網格感，完全不處理純黑)
        int d_a = 0;
        if (y_val <= 30) {
            d_a = 255; 
        } else if (y_val < 50) {
            d_a = ((50 - y_val) * 255) / 20; 
        }
        
        // 取三大保護機制的最高值
        int raw_weight = max({f_a, l_a, d_a});
        int enh_weight = 255 - raw_weight;

        // 最終 RGB 融合
        dst[i*3]   = (uchar)((src[i*3]   * raw_weight + b_enh * enh_weight) >> 8);
        dst[i*3+1] = (uchar)((src[i*3+1] * raw_weight + g_enh * enh_weight) >> 8);
        dst[i*3+2] = (uchar)((src[i*3+2] * raw_weight + r_enh * enh_weight) >> 8);
    }
}

// =========================================================================
//  PART F: 主程式 & 評估函式 (保持原樣)
// =========================================================================
class AsyncFrameReader {
private:
    VideoCapture cap; queue<Mat> frame_queue; mutex mtx; condition_variable cv_in, cv_out;
    thread worker; atomic<bool> stop_flag, finished; int max_queue_size;
    void loop() {
        while (!stop_flag) {
            Mat frame;
            if (!cap.read(frame)) { finished = true; cv_out.notify_all(); break; }
            unique_lock<mutex> lock(mtx);
            cv_in.wait(lock, [this]() { return frame_queue.size() < (size_t)max_queue_size || stop_flag.load(); });
            if (stop_flag) break;
            frame_queue.push(frame); lock.unlock(); cv_out.notify_one();
        }
    }
public:
    AsyncFrameReader(string path, int buffer_size = 60) : max_queue_size(buffer_size), stop_flag(false), finished(false) {
        if (isdigit(path[0])) cap.open(stoi(path)); else cap.open(path);
        if (cap.isOpened()) worker = thread(&AsyncFrameReader::loop, this);
    }
    ~AsyncFrameReader() { stop_flag = true; cv_in.notify_all(); if (worker.joinable()) worker.join(); cap.release(); }
    bool isOpened() { return cap.isOpened(); }
    int  get(int propId) { return (int)cap.get(propId); }
    bool getFrame(Mat& out) {
        unique_lock<mutex> lock(mtx);
        cv_out.wait(lock, [this]() { return !frame_queue.empty() || finished.load() || stop_flag.load(); });
        if (frame_queue.empty()) return false;
        out = frame_queue.front(); frame_queue.pop(); lock.unlock(); cv_in.notify_one(); return true;
    }
};

double compute_masked_entropy(const Mat& gray, const uchar* mask) {
    int hist[256] = {0}; int valid = 0;
    for(int r=0; r<gray.rows; r++) {
        const uchar* p = gray.ptr<uchar>(r);
        for(int c=0; c<gray.cols; c++) {
            if(mask[r*gray.cols+c] == 0) { hist[p[c]]++; valid++; }
        }
    }
    if(valid == 0) return 0; double ent = 0;
    for(int i=0; i<256; i++) {
        if(hist[i] > 0) { double p = (double)hist[i]/valid; ent -= p * log2(p); }
    }
    return ent;
}

double compute_masked_grad(const Mat& gray, const uchar* mask) {
    Mat gx, gy, grad; Sobel(gray, gx, CV_32F, 1, 0, 3); Sobel(gray, gy, CV_32F, 0, 1, 3); magnitude(gx, gy, grad);
    double sum = 0; int valid = 0;
    for(int r=0; r<grad.rows; r++) {
        const float* p = grad.ptr<float>(r);
        for(int c=0; c<grad.cols; c++) {
            if(mask[r*grad.cols+c] == 0) { sum += p[c]; valid++; }
        }
    }
    return valid > 0 ? (sum / valid) : 0;
}

int main(int argc, char** argv) {
    SetConsoleOutputCP(65001);

    // Image mode for Fire Scene AI Agent: --image input_path output_path
    if (argc >= 4 && string(argv[1]) == "--image") {
        Mat input = imread(argv[2], IMREAD_COLOR);
        if (input.empty()) {
            cerr << "Error: cannot read input image: " << argv[2] << endl;
            return 2;
        }
        Mat output(input.size(), input.type());
        vector<unsigned char> mask((size_t)input.cols * input.rows, 0);
        double t0 = (double)getTickCount();
        thesis_core_process(input.data, output.data, mask.data(), input.cols, input.rows);
        double elapsed_ms = ((double)getTickCount() - t0) / getTickFrequency() * 1000.0;
        if (!imwrite(argv[3], output)) {
            cerr << "Error: cannot write output image: " << argv[3] << endl;
            return 3;
        }
        cout << "Thesis enhancement completed. " << input.cols << "x" << input.rows
             << " | " << fixed << setprecision(2) << elapsed_ms << " ms" << endl;
        cout << "Output: " << argv[3] << endl;
        return 0;
    }
    string source = (argc > 1) ? argv[1] : "./firepov.mp4";
    AsyncFrameReader reader(source);
    if (!reader.isOpened()) { cout << "無法開啟影片" << endl; return -1; }

    int width  = reader.get(CAP_PROP_FRAME_WIDTH), height = reader.get(CAP_PROP_FRAME_HEIGHT);
    bool resize_flag = (width > 1280); Size proc_size = resize_flag ? Size(1280, 720) : Size(width, height);
    VideoWriter writer; string output_file = "result_clean_video_optimized_final_4_20260726.mp4";
    writer.open(output_file, VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, Size(proc_size.width * 2, proc_size.height));
    if (!writer.isOpened()) writer.open(output_file, VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, Size(proc_size.width * 2, proc_size.height));

    cout << "==========================================================================================" << endl;
    cout << "| Frame |  ms  | FPS | Gamma | OEnt | EEnt |+Ent| OGrad| EGrad|+Grad|" << endl;
    cout << "==========================================================================================" << endl;

    Mat raw_frame, frame, result, combined;
    unsigned char* frame_mask = (unsigned char*)malloc(proc_size.width * proc_size.height);
    int idx = 0;

    while (reader.getFrame(raw_frame)) {
        if (resize_flag) resize(raw_frame, frame, proc_size); else frame = raw_frame;
        result.create(frame.size(), frame.type());

        double t1 = (double)getTickCount();
        thesis_core_process(frame.data, result.data, frame_mask, frame.cols, frame.rows);
        double t2 = (double)getTickCount();
        double time_ms = ((t2-t1) / getTickFrequency()) * 1000.0, fps = 1000.0 / time_ms;

        Mat gray_ori, gray_enh; cvtColor(frame, gray_ori, COLOR_BGR2GRAY); cvtColor(result, gray_enh, COLOR_BGR2GRAY);
        double ori_ent = compute_masked_entropy(gray_ori, frame_mask), enh_ent = compute_masked_entropy(gray_enh, frame_mask);
        double ori_grad = compute_masked_grad(gray_ori, frame_mask), enh_grad = compute_masked_grad(gray_enh, frame_mask);

        if (idx % 10 == 0) {
            cout << "| " << setw(5) << idx << " | " << setw(4) << fixed << setprecision(1) << time_ms << " | " << setw(3) << (int)fps
                 << " | " << setw(4) << setprecision(2) << current_dynamic_gamma 
                 << " | " << setw(4) << setprecision(2) << ori_ent << " | " << setw(4) << enh_ent << " | " << setw(4) << setprecision(2) << (enh_ent - ori_ent)
                 << " | " << setw(4) << setprecision(1) << ori_grad << " | " << setw(4) << enh_grad << " | " << setw(4) << setprecision(1) << (enh_grad - ori_grad) << " |" << endl;
        }

        hconcat(frame, result, combined);
        putText(combined, "Original", Point(30,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2);
        putText(combined, "Ent:" + to_string(ori_ent).substr(0,4) + "  Grad:" + to_string((int)ori_grad), Point(30,80), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(180,180,180), 1);
        
        putText(combined, "Enhanced", Point(frame.cols+30,40), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,255,0), 2);
        putText(combined, "Ent:" + to_string(enh_ent).substr(0,4) + "(+" + to_string(enh_ent - ori_ent).substr(0,4) + ")" + "  Grad:" + to_string((int)enh_grad) + "(+" + to_string((int)(enh_grad - ori_grad)) + ")", Point(frame.cols+30,80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,220,180), 1);
        
        string perf_text = "Time:" + to_string(time_ms).substr(0,5) + "ms  FPS:" + to_string((int)fps) + "  Gamma:" + to_string(current_dynamic_gamma).substr(0,4);
        putText(combined, perf_text, Point(frame.cols+30,115), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(0,100,255), 2);

        if (writer.isOpened()) writer.write(combined);
        imshow("Real-Time Enhancement", combined);
        idx++; if (waitKey(1) == 27) break;
    }
    free(frame_mask); destroyAllWindows(); return 0;
}