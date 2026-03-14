#include <stdexcept>
#include <string>
#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

#include "../line2Dup.hpp"

namespace py = pybind11;

namespace {

cv::Mat from_numpy_u8(const py::array &array) {
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr(array);
    py::buffer_info info = arr.request();

    if (info.ndim == 2) {
        int h = static_cast<int>(info.shape[0]);
        int w = static_cast<int>(info.shape[1]);
        cv::Mat mat(h, w, CV_8UC1, info.ptr);
        return mat.clone();
    }

    if (info.ndim == 3) {
        int h = static_cast<int>(info.shape[0]);
        int w = static_cast<int>(info.shape[1]);
        int c = static_cast<int>(info.shape[2]);
        if (c == 1) {
            cv::Mat mat(h, w, CV_8UC1, info.ptr);
            return mat.clone();
        }
        if (c == 3) {
            cv::Mat mat(h, w, CV_8UC3, info.ptr);
            return mat.clone();
        }
    }

    throw std::runtime_error("只支持 uint8 的灰度(H,W)或彩色(H,W,3)图像");
}

std::vector<std::string> list_to_strings(const py::list &items) {
    std::vector<std::string> out;
    out.reserve(items.size());
    for (const auto &item : items) {
        out.push_back(py::cast<std::string>(item));
    }
    return out;
}

} // namespace

class PySubpixelDetector {
public:
    PySubpixelDetector(int num_features = 35,
                       const std::vector<int> &T = {4, 8},
                       float weak_thresh = 20.0f,
                       float strong_thresh = 40.0f)
        : detector_(num_features, T, weak_thresh, strong_thresh) {
        detector_.setStartAngle(-5.0f);
        detector_.setEndAngl(5.0f);
        detector_.setStepAngle(0.5f);
        detector_.setStartScale(1.0f);
        detector_.setEndScale(1.0f);
        detector_.setStepScale(1.0f);
        detector_.setIcp(false);
    }

    void set_angle_range(float start_angle, float end_angle, float step_angle) {
        detector_.setStartAngle(start_angle);
        detector_.setEndAngl(end_angle);
        detector_.setStepAngle(step_angle);
    }

    void set_scale_range(float start_scale, float end_scale, float step_scale) {
        detector_.setStartScale(start_scale);
        detector_.setEndScale(end_scale);
        detector_.setStepScale(step_scale);
    }

    void set_icp(bool enabled) {
        detector_.setIcp(enabled);
    }

    void create_model(const py::array &image,
                      const py::object &mask_obj = py::none(),
                      const py::object &invalid_mask_obj = py::none(),
                      const std::string &model_name = "classID_0") {
        cv::Mat src = from_numpy_u8(image);
        cv::Mat mask;
        cv::Mat invalid_mask;

        if (mask_obj.is_none()) {
            mask = cv::Mat(src.size(), CV_8UC1, cv::Scalar(255));
        } else {
            mask = from_numpy_u8(mask_obj.cast<py::array>());
        }

        if (invalid_mask_obj.is_none()) {
            // line2Dup uses this mask as the valid region for gradient features.
            // All-zero mask will suppress almost all template features.
            invalid_mask = cv::Mat(src.size(), CV_8UC1, cv::Scalar(255));
        } else {
            invalid_mask = from_numpy_u8(invalid_mask_obj.cast<py::array>());
        }

        if (mask.size() != src.size() || invalid_mask.size() != src.size()) {
            throw std::runtime_error("mask/invalid_mask 尺寸必须与 image 一致");
        }

        cv::Mat debug_out;
        detector_.createModel(src, debug_out, mask, invalid_mask, model_name, cv::Rect());
    }

    int save_model(const std::string &path) {
        return detector_.saveModel(path);
    }

    void write_classes(const std::string &path) {
        detector_.writeClasses(path);
    }

    int read_classes(const std::string &path) {
        std::vector<std::string> class_ids;
        return detector_.readClasses(path, class_ids);
    }

    py::dict match(const py::array &image,
                   float threshold = 90.0f,
                   const py::list &class_ids = py::list(),
                   const py::object &mask_obj = py::none(),
                   float start_angle = -360.0f,
                   float end_angle = 360.0f) {
        cv::Mat src = from_numpy_u8(image);
        cv::Mat mask;
        if (mask_obj.is_none()) {
            mask = cv::Mat();
        } else {
            mask = from_numpy_u8(mask_obj.cast<py::array>());
            if (mask.size() != src.size()) {
                throw std::runtime_error("mask 尺寸必须与 image 一致");
            }
        }

        std::vector<std::string> ids = list_to_strings(class_ids);
        std::map<std::string, std::vector<line2Dup::Match>> res =
            detector_.match(src, threshold, ids, mask, start_angle, end_angle);

        py::dict out;
        for (const auto &kv : res) {
            py::list item_list;
            for (const auto &m : kv.second) {
                py::dict md;
                md["x"] = m.x;
                md["y"] = m.y;
                md["similarity"] = m.similarity;
                md["template_id"] = m.template_id;
                md["angle"] = m.angle;
                md["class_id"] = m.class_id;
                item_list.append(md);
            }
            out[py::str(kv.first)] = item_list;
        }
        return out;
    }

private:
    line2Dup::Detector detector_;
};

PYBIND11_MODULE(sbm_subpixel, m) {
    m.doc() = "Python bindings for shape_based_matching-subpixel detector";

    py::class_<PySubpixelDetector>(m, "SubpixelDetector")
        .def(py::init<int, const std::vector<int>&, float, float>(),
             py::arg("num_features") = 35,
             py::arg("T") = std::vector<int>{4, 8},
             py::arg("weak_thresh") = 20.0f,
             py::arg("strong_thresh") = 40.0f)
        .def("set_angle_range", &PySubpixelDetector::set_angle_range,
             py::arg("start_angle"), py::arg("end_angle"), py::arg("step_angle"))
        .def("set_scale_range", &PySubpixelDetector::set_scale_range,
             py::arg("start_scale"), py::arg("end_scale"), py::arg("step_scale"))
        .def("set_icp", &PySubpixelDetector::set_icp, py::arg("enabled"))
        .def("create_model", &PySubpixelDetector::create_model,
             py::arg("image"),
             py::arg("mask") = py::none(),
             py::arg("invalid_mask") = py::none(),
             py::arg("model_name") = "classID_0")
        .def("save_model", &PySubpixelDetector::save_model, py::arg("path"))
        .def("write_classes", &PySubpixelDetector::write_classes, py::arg("path"))
        .def("read_classes", &PySubpixelDetector::read_classes, py::arg("path"))
        .def("match", &PySubpixelDetector::match,
             py::arg("image"),
             py::arg("threshold") = 90.0f,
             py::arg("class_ids") = py::list(),
             py::arg("mask") = py::none(),
             py::arg("start_angle") = -360.0f,
             py::arg("end_angle") = 360.0f);
}

