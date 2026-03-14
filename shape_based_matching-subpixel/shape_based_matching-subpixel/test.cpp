#include "line2Dup.hpp"
#include <algorithm>
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <direct.h>
#include <stdexcept>

#include <opencv2/core/utils/logger.hpp>

#include "cuda_icp/icp.h"

using namespace std;
using namespace cv;

static std::string prefix = ".\\test\\";

static constexpr float kCase0ScaleMin = 0.5f;
static constexpr float kCase0ScaleMax = 2.5f;
static constexpr float kCase0ScaleStep = 0.02f;
static constexpr float kCase0MatchThreshold = 45.0f;
static constexpr float kCase0NmsThreshold = 0.35f;

struct ScaleCaseConfig
{
    std::string case_name = "case0";
    std::string class_id = "cross";
    std::string train_image_path = prefix + "case0\\templ\\cross.png";
    std::string train_mask_path;
    std::string train_edge_mask_path;
    std::string templ_path = prefix + "case0\\cross_templ.yaml";
    std::string info_path = prefix + "case0\\cross_info.yaml";
    std::string test_img_path = prefix + "case0\\1.png";
    std::string result_path = prefix + "case0\\result\\1_runtime.png";
    int num_feature = 150;
    int stride = 32;
    int top_k = 5;
    int train_padding = 0;
    int min_template_width = 48;
    int min_template_height = 48;
    float angle_min = 0.0f;
    float angle_max = 0.0f;
    float angle_step = 1.0f;
    float scale_min = kCase0ScaleMin;
    float scale_max = kCase0ScaleMax;
    float scale_step = kCase0ScaleStep;
    float match_threshold = kCase0MatchThreshold;
    float nms_threshold = kCase0NmsThreshold;
};

static bool is_absolute_path(const std::string& path)
{
    if (path.size() >= 2 && path[1] == ':') return true;
    if (path.size() >= 2 && path[0] == '\\' && path[1] == '\\') return true;
    if (path.size() >= 2 && path[0] == '/' && path[1] == '/') return true;
    return false;
}

static std::string normalize_slashes(std::string path)
{
    std::replace(path.begin(), path.end(), '/', '\\');
    return path;
}

static std::string dirname_of(const std::string& path)
{
    const std::string normalized = normalize_slashes(path);
    const size_t pos = normalized.find_last_of('\\');
    return pos == std::string::npos ? "." : normalized.substr(0, pos);
}

static std::string join_path(const std::string& base, const std::string& leaf)
{
    if (leaf.empty()) return normalize_slashes(base);
    if (is_absolute_path(leaf)) return normalize_slashes(leaf);

    std::string normalized_base = normalize_slashes(base);
    std::string normalized_leaf = normalize_slashes(leaf);
    if (!normalized_base.empty() && normalized_base.back() != '\\') {
        normalized_base += "\\";
    }
    return normalized_base + normalized_leaf;
}

static void mkdir_recursive(const std::string& dir_path)
{
    std::string normalized = normalize_slashes(dir_path);
    if (normalized.empty() || normalized == ".") return;

    size_t start = 0;
    if (normalized.size() >= 2 && normalized[1] == ':') {
        start = 3;
    } else if (normalized.size() >= 2 && normalized[0] == '\\' && normalized[1] == '\\') {
        start = normalized.find('\\', 2);
        if (start == std::string::npos) return;
        start = normalized.find('\\', start + 1);
        if (start == std::string::npos) return;
        ++start;
    }

    for (size_t pos = start; pos <= normalized.size(); ++pos) {
        if (pos != normalized.size() && normalized[pos] != '\\') continue;
        const std::string piece = normalized.substr(0, pos);
        if (!piece.empty() && piece != "." && piece.back() != ':') {
            _mkdir(piece.c_str());
        }
    }
}

static std::string resolve_scale_config_path(const std::string& config_arg)
{
    if (config_arg.empty()) {
        return prefix + "case0\\config.yaml";
    }

    std::string normalized_arg = normalize_slashes(config_arg);
    if (normalized_arg.size() >= 5 &&
        (normalized_arg.rfind(".yaml") == normalized_arg.size() - 5 ||
         normalized_arg.rfind(".yml") == normalized_arg.size() - 4)) {
        return normalized_arg;
    }

    if (normalized_arg.find('\\') != std::string::npos || normalized_arg.find('/') != std::string::npos) {
        return join_path(normalized_arg, "config.yaml");
    }

    return join_path(prefix, normalized_arg + "\\config.yaml");
}

static ScaleCaseConfig default_scale_case_config()
{
    return ScaleCaseConfig();
}

static void resolve_scale_case_paths(ScaleCaseConfig& config, const std::string& config_path)
{
    const std::string config_dir = dirname_of(config_path);
    config.train_image_path = join_path(config_dir, config.train_image_path);
    if (!config.train_mask_path.empty()) config.train_mask_path = join_path(config_dir, config.train_mask_path);
    if (!config.train_edge_mask_path.empty()) config.train_edge_mask_path = join_path(config_dir, config.train_edge_mask_path);
    config.templ_path = join_path(config_dir, config.templ_path);
    config.info_path = join_path(config_dir, config.info_path);
    config.test_img_path = join_path(config_dir, config.test_img_path);
    config.result_path = join_path(config_dir, config.result_path);
}

static ScaleCaseConfig load_scale_case_config(const std::string& config_arg)
{
    const std::string config_path = resolve_scale_config_path(config_arg);
    ScaleCaseConfig config = default_scale_case_config();

    FileStorage fs(config_path, FileStorage::READ);
    if (!fs.isOpened()) {
        if (config_arg.empty()) {
            resolve_scale_case_paths(config, config_path);
            return config;
        }

        std::cerr << "failed to open scale config: " << config_path << std::endl;
        throw std::runtime_error("scale config not found");
    }

    if (!fs["case_name"].empty()) fs["case_name"] >> config.case_name;
    if (!fs["class_id"].empty()) fs["class_id"] >> config.class_id;
    if (!fs["train_image"].empty()) fs["train_image"] >> config.train_image_path;
    if (!fs["train_mask"].empty()) fs["train_mask"] >> config.train_mask_path;
    if (!fs["train_edge_mask"].empty()) fs["train_edge_mask"] >> config.train_edge_mask_path;
    if (!fs["template_file"].empty()) fs["template_file"] >> config.templ_path;
    if (!fs["info_file"].empty()) fs["info_file"] >> config.info_path;
    if (!fs["test_image"].empty()) fs["test_image"] >> config.test_img_path;
    if (!fs["result_image"].empty()) fs["result_image"] >> config.result_path;
    if (!fs["num_feature"].empty()) fs["num_feature"] >> config.num_feature;
    if (!fs["stride"].empty()) fs["stride"] >> config.stride;
    if (!fs["top_k"].empty()) fs["top_k"] >> config.top_k;
    if (!fs["train_padding"].empty()) fs["train_padding"] >> config.train_padding;
    if (!fs["min_template_width"].empty()) fs["min_template_width"] >> config.min_template_width;
    if (!fs["min_template_height"].empty()) fs["min_template_height"] >> config.min_template_height;
    if (!fs["angle_min"].empty()) fs["angle_min"] >> config.angle_min;
    if (!fs["angle_max"].empty()) fs["angle_max"] >> config.angle_max;
    if (!fs["angle_step"].empty()) fs["angle_step"] >> config.angle_step;
    if (!fs["scale_min"].empty()) fs["scale_min"] >> config.scale_min;
    if (!fs["scale_max"].empty()) fs["scale_max"] >> config.scale_max;
    if (!fs["scale_step"].empty()) fs["scale_step"] >> config.scale_step;
    if (!fs["match_threshold"].empty()) fs["match_threshold"] >> config.match_threshold;
    if (!fs["nms_threshold"].empty()) fs["nms_threshold"] >> config.nms_threshold;
    fs.release();

    resolve_scale_case_paths(config, config_path);
    return config;
}

static cv::Mat load_binary_mask(const std::string& path, const cv::Size& expected_size)
{
    if (path.empty()) return cv::Mat();

    cv::Mat raw = imread(path, IMREAD_GRAYSCALE);
    if (raw.empty()) {
        throw std::runtime_error("failed to load mask image: " + path);
    }
    if (raw.size() != expected_size) {
        throw std::runtime_error("mask size mismatch: " + path);
    }

    cv::Mat mask;
    cv::threshold(raw, mask, 0, 255, THRESH_BINARY);
    return mask;
}

static cv::Mat alpha_mask_from_image(const cv::Mat& img)
{
    if (img.channels() != 4) return cv::Mat();

    std::vector<cv::Mat> channels;
    split(img, channels);
    double min_val = 0.0;
    double max_val = 0.0;
    minMaxLoc(channels[3], &min_val, &max_val);
    if (min_val == 255.0 && max_val == 255.0) {
        return cv::Mat();
    }

    cv::Mat mask;
    cv::threshold(channels[3], mask, 0, 255, THRESH_BINARY);
    return mask;
}

static cv::Mat bgr_from_image(const cv::Mat& img)
{
    if (img.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(img, bgr, COLOR_BGRA2BGR);
        return bgr;
    }
    return img.clone();
}

static cv::Mat pad_image(const cv::Mat& img, int padding, const cv::Scalar& fill)
{
    if (padding <= 0) return img.clone();

    cv::Mat padded(img.rows + 2 * padding, img.cols + 2 * padding, img.type(), fill);
    img.copyTo(padded(Rect(padding, padding, img.cols, img.rows)));
    return padded;
}

// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

void scale_test(const ScaleCaseConfig& config, string mode = "test", bool viewResult = false){
    line2Dup::Detector detector(config.num_feature, { 4, 8 });

    if(mode == "train"){
        Mat raw_img = imread(config.train_image_path, IMREAD_UNCHANGED);
        assert(!raw_img.empty() && "check your img path");

        Mat img = bgr_from_image(raw_img);
        Mat mask = load_binary_mask(config.train_mask_path, img.size());
        if (mask.empty()) {
            mask = alpha_mask_from_image(raw_img);
        }
        if (mask.empty()) {
            mask = Mat(img.size(), CV_8UC1, Scalar(255));
        }

        Mat edge_mask = load_binary_mask(config.train_edge_mask_path, img.size());
        if (edge_mask.empty()) {
            edge_mask = mask.clone();
        }

        img = pad_image(img, config.train_padding, Scalar::all(0));
        mask = pad_image(mask, config.train_padding, Scalar::all(0));
        edge_mask = pad_image(edge_mask, config.train_padding, Scalar::all(0));

        shape_based_matching::shapeInfo_producer shapes(img, mask, edge_mask);
        if (std::abs(config.angle_max - config.angle_min) > 1e-6f) {
            shapes.angle_range = { config.angle_min, config.angle_max };
            shapes.angle_step = config.angle_step;
        } else if (std::abs(config.angle_min) > 1e-6f) {
            shapes.angle_range = { config.angle_min };
        }
        if (std::abs(config.scale_max - config.scale_min) > 1e-6f) {
            shapes.scale_range = { config.scale_min, config.scale_max };
        } else {
            shapes.scale_range = { config.scale_min };
        }
        shapes.scale_step = config.scale_step;
        shapes.produce_infos();

        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        for (auto& info : shapes.infos) {
            int templ_id = detector.addTemplate(
                shapes.src_of(info),
                config.class_id,
                shapes.mask_of(info),
                shapes.edgeMask_of(info),
                int(config.num_feature * info.scale));
            std::cout << "templ_id: " << templ_id << std::endl;
            if (templ_id != -1) {
                infos_have_templ.push_back(info);
            }
        }

        mkdir_recursive(dirname_of(config.templ_path));
        mkdir_recursive(dirname_of(config.info_path));
        detector.writeClasses(config.templ_path);
        shapes.save_infos(infos_have_templ, config.info_path);
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode == "test"){
        std::vector<std::string> ids;
        ids.push_back(config.class_id);
        detector.readClasses(config.templ_path, ids);
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(config.info_path);

        Mat test_img = imread(config.test_img_path);
        assert(!test_img.empty() && "check your img path");

        int stride = config.stride;
        int n = test_img.rows / stride;
        int m = test_img.cols / stride;
        Rect roi(0, 0, stride * m, stride * n);
        Mat img = test_img(roi).clone();
        assert(img.isContinuous());

        std::cout << config.case_name << " matching..." << std::endl;
        std::cout << "class_id: " << config.class_id << std::endl;
        std::cout << "threshold: " << config.match_threshold
                  << ", angle range: [" << config.angle_min << ", " << config.angle_max << "]"
                  << ", angle step: " << config.angle_step
                  << ", scale range: [" << config.scale_min << ", " << config.scale_max << "]"
                  << ", step: " << config.scale_step << std::endl;

        Timer timer;
        auto match_map = detector.match(img, config.match_threshold, ids);
        std::vector<line2Dup::Match> matches;
        auto match_it = match_map.find(config.class_id);
        if (match_it != match_map.end()) {
            matches = match_it->second;
        }
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;

        std::vector<line2Dup::Match> filtered_matches;
        std::vector<Rect> boxes;
        std::vector<float> scores;
        filtered_matches.reserve(matches.size());
        boxes.reserve(matches.size());
        scores.reserve(matches.size());
        for (const auto& match : matches) {
            auto templ = detector.getTemplates(config.class_id, match.template_id);
            if (templ[0].width < config.min_template_width || templ[0].height < config.min_template_height) {
                continue;
            }
            filtered_matches.push_back(match);
            boxes.emplace_back(match.x, match.y, templ[0].width, templ[0].height);
            scores.push_back(match.similarity);
        }

        std::cout << "matches after size filter: " << filtered_matches.size() << std::endl;

        std::vector<int> keep;
        cv_dnn::NMSBoxes(boxes, scores, config.match_threshold, config.nms_threshold, keep);
        std::cout << "matches after nms: " << keep.size() << std::endl;

        size_t top5 = std::min<size_t>(size_t(config.top_k), keep.size());

        Mat result_img = img.clone();
        for(size_t i = 0; i < top5; i++){
            auto match = filtered_matches[keep[i]];
            auto templ = detector.getTemplates(config.class_id, match.template_id);

            int x = templ[0].width / 2 + match.x;
            int y = templ[0].height / 2 + match.y;
            int r = templ[0].width / 2;
            Scalar color(255, rand() % 255, rand() % 255);

            std::cout << "top" << i + 1
                      << ": score=" << match.similarity
                      << ", x=" << match.x
                      << ", y=" << match.y
                      << ", w=" << templ[0].width
                      << ", h=" << templ[0].height;
            if (match.template_id >= 0 && match.template_id < infos.size()) {
                std::cout << ", angle=" << infos[match.template_id].angle
                          << ", scale=" << infos[match.template_id].scale;
            }
            std::cout
                      << std::endl;

            cv::putText(result_img, to_string(int(round(match.similarity))),
                        Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2, color);
            cv::rectangle(result_img, Rect(match.x, match.y, templ[0].width, templ[0].height), color, 2);
        }

        mkdir_recursive(dirname_of(config.result_path));
        bool saved = imwrite(config.result_path, result_img);
        std::cout << "result image: " << (saved ? config.result_path : "save failed") << std::endl;
        if(viewResult){
            imshow("img", result_img);
            waitKey(0);
        }

        std::cout << "test end" << std::endl << std::endl;
    }
}

void angle_test(string mode = "test", bool viewICP = false){
    //line2Dup::Detector detector(128, {4, 8});
	line2Dup::Detector detector(35, { 4, 8 });

    //mode = "train";
    if(mode == "train")
	{
        Mat img = imread(prefix+"case1\\train.tif");
        assert(!img.empty() && "check your img path");

        Rect roi(130, 110, 270, 270);
		roi = cv::Rect(0, 0, img.cols, img.rows);
        img = img(roi).clone();
        Mat mask = Mat(img.size(), CV_8UC1, cv::Scalar(255));

        // padding to avoid rotating out
        int padding = 100;
        cv::Mat padded_img = cv::Mat(img.rows + 2*padding, img.cols + 2*padding, img.type(), cv::Scalar::all(0));
        img.copyTo(padded_img(Rect(padding, padding, img.cols, img.rows)));

        cv::Mat padded_mask = cv::Mat(mask.rows + 2*padding, mask.cols + 2*padding, mask.type(), cv::Scalar::all(0));
        mask.copyTo(padded_mask(Rect(padding, padding, img.cols, img.rows)));

        shape_based_matching::shapeInfo_producer shapes(padded_img, padded_mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for(auto& info: shapes.infos)
		{
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info), shapes.edgeMask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"case1\\test_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case1\\test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }
	else if(mode=="test")
	{
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(prefix+"case1\\test_templ.yaml", ids);

        // angle & scale are saved here, fetched by match id
        auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "case1\\test_info.yaml");

        Mat test_img = imread(prefix+"case1\\test.tif");
        assert(!test_img.empty() && "check your img path");

        int padding = 100;
        cv::Mat padded_img = cv::Mat(test_img.rows + 2*padding,
                                     test_img.cols + 2*padding, test_img.type(), cv::Scalar::all(0));
        test_img.copyTo(padded_img(Rect(padding, padding, test_img.cols, test_img.rows)));

        int stride = 16;
        int n = padded_img.rows/stride;
        int m = padded_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);
        Mat img = padded_img(roi).clone();
        assert(img.isContinuous());

//        cvtColor(img, img, CV_BGR2GRAY);

        std::cout << "test img size: " << img.rows * img.cols << std::endl << std::endl;

        Timer timer;
        auto match_map = detector.match(img, 40, ids);
        std::vector<line2Dup::Match> matches;
        auto match_it = match_map.find("test");
        if (match_it != match_map.end()) {
            matches = match_it->second;
        }
        timer.out();


        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 5;
        if(top5>matches.size()) top5=matches.size();

        // construct scene
        Scene_edge scene;
        // buffer
        vector<::Vec2f> pcd_buffer, normal_buffer;
        scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

		
        if(img.channels() == 1) cvtColor(img, img, cv::COLOR_GRAY2BGR);

        cv::Mat edge_global;  // get edge
        {
            cv::Mat gray;
            if(img.channels() > 1){
                cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
            }else{
                gray = img;
            }

            cv::Mat smoothed = gray;
            cv::Canny(smoothed, edge_global, 30, 60);

            if(edge_global.channels() == 1) cvtColor(edge_global, edge_global, cv::COLOR_GRAY2BGR);
        }

		int imageWidth = 468;

        Mat result_img = edge_global.clone();
        for(int i=top5-1; i>=0; i--)
        {
            Mat edge = edge_global.clone();

            auto match = matches[i];
            auto templ = detector.getTemplates("test",
                                               match.template_id);

            // 270 is width of template image
            // 100 is padding when training
            // tl_x/y: template croping topleft corner when training

            float r_scaled = imageWidth /2.0f*infos[match.template_id].scale;

            // scaling won't affect this, because it has been determined by warpAffine
            // cv::warpAffine(src, dst, rot_mat, src.size()); last param
            float train_img_half_width = imageWidth /2.0f + 100;

            // center x,y of train_img in test img
            float x =  match.x - templ[0].tl_x + train_img_half_width;
            float y =  match.y - templ[0].tl_y + train_img_half_width;

            vector<::Vec2f> model_pcd(templ[0].features.size());
            for(int i=0; i<templ[0].features.size(); i++){
                auto& feat = templ[0].features[i];
                model_pcd[i] = {
                    float(feat.x + match.x),
                    float(feat.y + match.y)
                };
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            cv::Vec3b randColor;
            randColor[0] = 0;
            randColor[1] = 0;
            randColor[2] = 255;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(edge, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
                cv::circle(result_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            if(viewICP){
                imshow("icp", edge);
                waitKey(0);
            }


            randColor[0] = 0;
            randColor[1] = 255;
            randColor[2] = 0;
            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                float x = feat.x + match.x;
                float y = feat.y + match.y;
                float new_x = result.transformation_[0][0]*x + result.transformation_[0][1]*y + result.transformation_[0][2];
                float new_y = result.transformation_[1][0]*x + result.transformation_[1][1]*y + result.transformation_[1][2];

                cv::circle(edge, {int(new_x+0.5f), int(new_y+0.5f)}, 2, randColor, -1);
                cv::circle(result_img, {int(new_x+0.5f), int(new_y+0.5f)}, 2, randColor, -1);
            }
            if(viewICP){
                imshow("icp", edge);
                waitKey(0);
            }

            double init_angle = infos[match.template_id].angle;
            init_angle = init_angle >= 180 ? (init_angle-360) : init_angle;

            double ori_diff_angle = std::abs(init_angle);
            double icp_diff_angle = std::abs(-std::asin(result.transformation_[1][0])/CV_PI*180 +
                    init_angle);
            double improved_angle = ori_diff_angle - icp_diff_angle;

            std::cout << "\n---------------" << std::endl;
			std::cout << "origin angle: " << ori_diff_angle << "  affine angle: " << icp_diff_angle << std::endl;
            std::cout << "init diff angle: " << ori_diff_angle << std::endl;
            std::cout << "improved angle: " << improved_angle << std::endl;
            std::cout << "match.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imwrite(prefix + "case1\\result.png", result_img);

        std::cout << "test end" << std::endl << std::endl;
		
    }
}

void noise_test(string mode = "test", bool viewResult = false){
    line2Dup::Detector detector(30, { 4, 8 });
    // Case 2 is a dense poster image; keep a stricter threshold to suppress text/background false positives.
    const float match_threshold = 75.0f;

    if (mode == "train")
    {
        Mat img = imread(prefix + "case2\\train.png");
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, cv::Scalar(255));

        shape_based_matching::shapeInfo_producer shapes(img, mask);
        shapes.angle_range = { 0, 360 };
        shapes.angle_step = 1;
        shapes.produce_infos();

        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = "test";
        for (auto& info : shapes.infos)
        {
            imshow("train", shapes.src_of(info));
            waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info), shapes.edgeMask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if (templ_id != -1) {
                infos_have_templ.push_back(info);
            }
        }

        detector.writeClasses(prefix + "case2\\test_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "case2\\test_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }
    else if (mode == "test")
    {
        std::vector<std::string> ids;
        ids.push_back("test");
        detector.readClasses(prefix + "case2\\test_templ.yaml", ids);

        Mat test_img = imread(prefix + "case2\\test.png");
        assert(!test_img.empty() && "check your img path");

        int stride = 16;
        int n = test_img.rows / stride;
        int m = test_img.cols / stride;
        Rect roi(0, 0, stride * m, stride * n);
        test_img = test_img(roi).clone();

        std::cout << "case2 matching..." << std::endl;
        std::cout << "Debug|x64 can take several minutes here." << std::endl;

        Timer timer;
        auto match_map = detector.match(test_img, match_threshold, ids);
        std::vector<line2Dup::Match> matches;
        auto match_it = match_map.find("test");
        if (match_it != match_map.end()) {
            matches = match_it->second;
        }
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for (const auto& match : matches) {
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates("test", match.template_id);
            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }

        if (!boxes.empty()) {
            cv_dnn::NMSBoxes(boxes, scores, match_threshold, 0.5f, idxs);
        }

        Mat result_img = test_img.clone();
        for (const auto idx : idxs) {
            auto match = matches[idx];
            auto templ = detector.getTemplates("test", match.template_id);

            int x = templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width / 2;
            cv::Vec3b randColor;
            randColor[0] = rand() % 155 + 100;
            randColor[1] = rand() % 155 + 100;
            randColor[2] = rand() % 155 + 100;

            for (int i = 0; i < templ[0].features.size(); i++) {
                auto feat = templ[0].features[i];
                cv::circle(result_img, { feat.x + match.x, feat.y + match.y }, 2, randColor, -1);
            }

            cv::putText(result_img, to_string(int(round(match.similarity))),
                Point(match.x + r - 10, match.y - 3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(result_img, { match.x, match.y }, { x, y }, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }

        imwrite(prefix + "case2\\result\\result.png", result_img);
        if (viewResult) {
            imshow("img", result_img);
            waitKey(0);
        }

        std::cout << "test end" << std::endl << std::endl;
    }
}

void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
        std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
        std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}

static void print_usage(const char* exe_name)
{
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << exe_name << " train_case0" << std::endl;
    std::cout << "  " << exe_name << " test_case0" << std::endl;
    std::cout << "  " << exe_name << " test_case0_view" << std::endl;
    std::cout << "  " << exe_name << " train_scale [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " test_scale [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " test_scale_view [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " train_affine [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " test_affine [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " test_affine_view [case_dir|config.yaml]" << std::endl;
    std::cout << "  " << exe_name << " train_case1" << std::endl;
    std::cout << "  " << exe_name << " test_case1" << std::endl;
    std::cout << "  " << exe_name << " test_case1_view" << std::endl;
    std::cout << "  " << exe_name << " train_case2" << std::endl;
    std::cout << "  " << exe_name << " test_case2" << std::endl;
    std::cout << "  " << exe_name << " test_case2_view" << std::endl;
}

int main(int argc, char** argv){
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    MIPP_test();

    std::string mode = argc > 1 ? argv[1] : "test_case1";
    std::string scale_config_arg = argc > 2 ? argv[2] : "";
    try
    {
        if (mode == "train_case0")
        {
            scale_test(load_scale_case_config(""), "train", false);
        }
        else if (mode == "test_case0")
        {
            scale_test(load_scale_case_config(""), "test", false);
        }
        else if (mode == "test_case0_view")
        {
            scale_test(load_scale_case_config(""), "test", true);
        }
        else if (mode == "train_scale" || mode == "train_affine")
        {
            scale_test(load_scale_case_config(scale_config_arg), "train", false);
        }
        else if (mode == "test_scale" || mode == "test_affine")
        {
            scale_test(load_scale_case_config(scale_config_arg), "test", false);
        }
        else if (mode == "test_scale_view" || mode == "test_affine_view")
        {
            scale_test(load_scale_case_config(scale_config_arg), "test", true);
        }
        else if (mode == "train_case1")
        {
            angle_test("train", false);
        }
        else if (mode == "test_case1")
        {
            angle_test("test", false);
        }
        else if (mode == "test_case1_view")
        {
            angle_test("test", true);
        }
        else if (mode == "train_case2")
        {
            noise_test("train", false);
        }
        else if (mode == "test_case2")
        {
            noise_test("test", false);
        }
        else if (mode == "test_case2_view")
        {
            noise_test("test", true);
        }
        else
        {
            print_usage(argv[0]);
            return 1;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
