#include <cmath>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void validate_shapes(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query_colors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query_errors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& train_colors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& train_errors
) {
    if (query_colors.ndim() != 2 || query_errors.ndim() != 2 ||
        train_colors.ndim() != 2 || train_errors.ndim() != 2) {
        throw std::invalid_argument("All inputs must be 2D arrays");
    }

    if (query_colors.shape(0) != query_errors.shape(0) ||
        query_colors.shape(1) != query_errors.shape(1)) {
        throw std::invalid_argument(
            "query_colors and query_errors must have matching shapes"
        );
    }

    if (train_colors.shape(0) != train_errors.shape(0) ||
        train_colors.shape(1) != train_errors.shape(1)) {
        throw std::invalid_argument(
            "train_colors and train_errors must have matching shapes"
        );
    }

    if (query_colors.shape(1) != train_colors.shape(1)) {
        throw std::invalid_argument(
            "Query and train arrays must have the same number of features"
        );
    }
}

py::array_t<float> cpp_chi_square_distance(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query_colors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& query_errors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& train_colors,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& train_errors,
    float error_floor = 1e-10f
) {
    validate_shapes(query_colors, query_errors, train_colors, train_errors);

    const py::ssize_t n_queries = query_colors.shape(0);
    const py::ssize_t n_train = train_colors.shape(0);
    const py::ssize_t n_features = query_colors.shape(1);

    auto query_colors_view = query_colors.unchecked<2>();
    auto query_errors_view = query_errors.unchecked<2>();
    auto train_colors_view = train_colors.unchecked<2>();
    auto train_errors_view = train_errors.unchecked<2>();

    py::array_t<float> distances({n_queries, n_train});
    auto distances_view = distances.mutable_unchecked<2>();

    py::gil_scoped_release release;

    for (py::ssize_t i = 0; i < n_queries; ++i) {
        for (py::ssize_t j = 0; j < n_train; ++j) {
            float distance = 0.0f;
            for (py::ssize_t k = 0; k < n_features; ++k) {
                const float color_diff = query_colors_view(i, k) - train_colors_view(j, k);
                const float denom = query_errors_view(i, k) * query_errors_view(i, k) +
                                    train_errors_view(j, k) * train_errors_view(j, k) +
                                    error_floor;
                distance += (color_diff * color_diff) / denom;
            }
            distances_view(i, j) = distance;
        }
    }

    return distances;
}

}  // namespace

PYBIND11_MODULE(cpp_chi_square_distance, m) {
    m.doc() = "C++ batched chi-square distance for KNN photo-z";
    m.def(
        "cpp_chi_square_distance",
        &cpp_chi_square_distance,
        py::arg("query_colors"),
        py::arg("query_errors"),
        py::arg("train_colors"),
        py::arg("train_errors"),
        py::arg("error_floor") = 1e-10f
    );
}
