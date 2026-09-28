// mcl_native.cpp: compiled version of the heavy maths in bot/localisation.py's
// MCL (motion update, likelihood-field weighting, resampling). The Python
// class keeps everything else: orchestration, RNG seeding, the IMU heading
// prior and the global-search jitter. Only the per-tick work that scales
// with particles x scan points x wall segments lives here, the same split
// as lidar_native.cpp. If this isn't built, MCL falls back to numpy
// (parity tested in native/test_mcl_native.py).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <random>
#include <vector>

namespace py = pybind11;

// One nearest-wall-segment distance, matching FieldModel.
// closest_point_on_segment / nearest_wall_batch's point-to-segment
// projection exactly (clamp t to [0,1], distance to the clamped point).
static inline double nearest_wall_dist(double px, double py,
                                       const std::vector<double> &seg_ax,
                                       const std::vector<double> &seg_ay,
                                       const std::vector<double> &seg_ex,
                                       const std::vector<double> &seg_ey,
                                       const std::vector<double> &seg_len2) {
    double best = 1e300;
    size_t n = seg_ax.size();
    for (size_t s = 0; s < n; ++s) {
        double rx = px - seg_ax[s], ry = py - seg_ay[s];
        double t = (rx * seg_ex[s] + ry * seg_ey[s]) / seg_len2[s];
        if (t < 0.0) t = 0.0;
        else if (t > 1.0) t = 1.0;
        double fx = seg_ax[s] + t * seg_ex[s];
        double fy = seg_ay[s] + t * seg_ey[s];
        double dx = px - fx, dy = py - fy;
        double d2 = dx * dx + dy * dy;
        if (d2 < best) best = d2;
    }
    return std::sqrt(best);
}

// Motion update, in place: particles is (N,3) [x, y, theta_deg], each
// particle's robot-frame (fwd, right) sample rotated into its own
// heading (see bot/localisation.py's comment on _motion_update for why
// per-particle, not one shared rotation). Field-clips x/y same as the
// Python version.
static void motion_update(py::array_t<double> particles,
                          double fwd, double right, double dth,
                          double trans_noise, double turn_noise,
                          double field_x, double field_y,
                          uint64_t seed) {
    auto buf = particles.mutable_unchecked<2>();
    py::ssize_t n = buf.shape(0);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> fwd_dist(fwd, trans_noise);
    std::normal_distribution<double> right_dist(right, trans_noise);
    std::normal_distribution<double> dth_dist(dth, turn_noise);

    for (py::ssize_t i = 0; i < n; ++i) {
        double th = buf(i, 2) * (M_PI / 180.0);
        double fwd_n = fwd_dist(rng);
        double right_n = right_dist(rng);
        double dth_n = dth_dist(rng);
        double ch = std::cos(th), sh = std::sin(th);
        double dx = ch * right_n + sh * fwd_n;
        double dy = ch * fwd_n - sh * right_n;
        double x = buf(i, 0) + dx;
        double y = buf(i, 1) + dy;
        if (x < -50.0) x = -50.0; else if (x > field_x + 50.0) x = field_x + 50.0;
        if (y < -50.0) y = -50.0; else if (y > field_y + 50.0) y = field_y + 50.0;
        buf(i, 0) = x;
        buf(i, 1) = y;
        double newh = std::fmod(buf(i, 2) + dth_n, 360.0);
        if (newh < 0.0) newh += 360.0;
        buf(i, 2) = newh;
    }
}

// Likelihood-field sensor weights: for each particle, transform every
// scan point (robot-frame) into field-frame using that particle's own
// pose, look up nearest-wall distance, and weight by a Gaussian on the
// mean squared distance (clamped). Matches bot/localisation.py's own
// _sensor_weights exactly, minus the IMU-prior multiply (still applied
// in Python afterward, on the returned array, since it's cheap and
// keeps the prior's logic in one place).
static py::array_t<double> sensor_weights(
        py::array_t<double> particles,
        py::array_t<double> points_local,
        py::array_t<double> seg_a, py::array_t<double> seg_ex_arr,
        py::array_t<double> seg_ey_arr, double sigma_mm) {
    auto p = particles.unchecked<2>();
    auto pts = points_local.unchecked<2>();
    auto sa = seg_a.unchecked<2>();
    auto sex = seg_ex_arr.unchecked<1>();
    auto sey = seg_ey_arr.unchecked<1>();

    py::ssize_t n_particles = p.shape(0);
    py::ssize_t n_pts = pts.shape(0);
    py::ssize_t n_segs = sex.shape(0);

    std::vector<double> seg_ax(n_segs), seg_ay(n_segs), seg_ex(n_segs), seg_ey(n_segs), seg_len2(n_segs);
    for (py::ssize_t s = 0; s < n_segs; ++s) {
        seg_ax[s] = sa(s, 0);
        seg_ay[s] = sa(s, 1);
        seg_ex[s] = sex(s);
        seg_ey[s] = sey(s);
        double l2 = seg_ex[s] * seg_ex[s] + seg_ey[s] * seg_ey[s];
        seg_len2[s] = l2 < 1e-9 ? 1e-9 : l2;
    }

    auto weights = py::array_t<double>(n_particles);
    auto w = weights.mutable_unchecked<1>();

    double cap = (4.0 * sigma_mm) * (4.0 * sigma_mm);
    double log_max = -1e300;
    std::vector<double> log_w(n_particles);

    for (py::ssize_t i = 0; i < n_particles; ++i) {
        double x = p(i, 0), y = p(i, 1);
        double th = p(i, 2) * (M_PI / 180.0);
        double c = std::cos(th), s = std::sin(th);
        double sum_sq = 0.0;
        for (py::ssize_t j = 0; j < n_pts; ++j) {
            double xl = pts(j, 0), yl = pts(j, 1);
            double px = x + xl * c + yl * s;
            double py_ = y - xl * s + yl * c;
            double d = nearest_wall_dist(px, py_, seg_ax, seg_ay, seg_ex, seg_ey, seg_len2);
            double d2 = d * d;
            if (d2 > cap) d2 = cap;
            sum_sq += d2;
        }
        double mean_sq = n_pts > 0 ? sum_sq / static_cast<double>(n_pts) : 0.0;
        double lw = -mean_sq / (2.0 * sigma_mm * sigma_mm);
        log_w[i] = lw;
        if (lw > log_max) log_max = lw;
    }

    double total = 0.0;
    for (py::ssize_t i = 0; i < n_particles; ++i) {
        double wi = std::exp(log_w[i] - log_max);
        w(i) = wi;
        total += wi;
    }
    if (total <= 1e-300) {
        double uni = 1.0 / static_cast<double>(n_particles);
        for (py::ssize_t i = 0; i < n_particles; ++i) w(i) = uni;
    } else {
        for (py::ssize_t i = 0; i < n_particles; ++i) w(i) /= total;
    }
    return weights;
}

// Low-variance (systematic) resampling, matching bot/localisation.py's own
// _resample exactly (same single-random-offset systematic scheme).
static py::array_t<double> resample(py::array_t<double> particles,
                                    py::array_t<double> weights,
                                    double u0) {
    auto p = particles.unchecked<2>();
    auto w = weights.unchecked<1>();
    py::ssize_t n = p.shape(0);

    std::vector<double> cumsum(n);
    double running = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        running += w(i);
        cumsum[i] = running;
    }
    cumsum[n - 1] = 1.0; // guard float rounding, matches the Python version

    auto out = py::array_t<double>({n, static_cast<py::ssize_t>(3)});
    auto o = out.mutable_unchecked<2>();
    py::ssize_t seg = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        double target = (u0 + static_cast<double>(i)) / static_cast<double>(n);
        while (seg < n - 1 && cumsum[seg] < target) {
            ++seg;
        }
        o(i, 0) = p(seg, 0);
        o(i, 1) = p(seg, 1);
        o(i, 2) = p(seg, 2);
    }
    return out;
}

PYBIND11_MODULE(mcl_native, m) {
    m.doc() = "C++ core for bot/localisation.py's MCL (motion update, "
              "likelihood-field sensor weights, resampling).";
    m.def("motion_update", &motion_update, "In-place per-particle odometry motion update.");
    m.def("sensor_weights", &sensor_weights, "Likelihood-field sensor weights for every particle.");
    m.def("resample", &resample, "Low-variance systematic resampling.");
}
