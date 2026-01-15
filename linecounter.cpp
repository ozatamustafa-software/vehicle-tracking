#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>

namespace py = pybind11;

class LineCounterCpp {
public:
    LineCounterCpp(int line_y, int offset=8, std::string direction="both", int cooldown_frames=12)
        : line_y_(line_y), offset_(offset), direction_(std::move(direction)), cooldown_frames_(cooldown_frames),
          total_(0), up_(0), down_(0) {}

    int total() const { return total_; }
    int up() const { return up_; }
    int down() const { return down_; }

    int update(const std::vector<std::tuple<int, float, float>>& id_centroids) {
        // cooldown azalt
        std::vector<int> to_del;
        to_del.reserve(cooldown_.size());
        for (auto &kv : cooldown_) {
            kv.second -= 1;
            if (kv.second <= 0) to_del.push_back(kv.first);
        }
        for (int id : to_del) cooldown_.erase(id);

        for (const auto& t : id_centroids) {
            int obj_id = std::get<0>(t);
            float cy = std::get<2>(t);

            auto it_prev = prev_y_.find(obj_id);
            float prev = 0.0f;
            bool has_prev = (it_prev != prev_y_.end());
            prev_y_[obj_id] = cy;

            if (!has_prev) continue;                 // ilk kez gördük → sayma yok
            if (cooldown_.find(obj_id) != cooldown_.end()) continue; // cooldown

            if (!in_band(cy)) continue;

            bool moved_down = (cy > prev);
            bool moved_up   = (cy < prev);

            if (direction_ == "down" && !moved_down) continue;
            if (direction_ == "up"   && !moved_up)   continue;

            total_ += 1;
            if (moved_down) down_ += 1;
            else if (moved_up) up_ += 1;

            cooldown_[obj_id] = cooldown_frames_;
        }

        return total_;
    }

private:
    bool in_band(float y) const {
        return (line_y_ - offset_) <= y && y <= (line_y_ + offset_);
    }

    int line_y_;
    int offset_;
    std::string direction_;
    int cooldown_frames_;

    int total_;
    int up_;
    int down_;

    std::unordered_map<int, float> prev_y_;
    std::unordered_map<int, int> cooldown_;
};

PYBIND11_MODULE(linecounter_cpp, m) {
    py::class_<LineCounterCpp>(m, "LineCounterCpp")
        .def(py::init<int,int,std::string,int>(),
             py::arg("line_y"),
             py::arg("offset")=8,
             py::arg("direction")="both",
             py::arg("cooldown_frames")=12)
        .def("update", &LineCounterCpp::update, py::arg("id_centroids"))
        .def_property_readonly("total", &LineCounterCpp::total)
        .def_property_readonly("up", &LineCounterCpp::up)
        .def_property_readonly("down", &LineCounterCpp::down);
}
