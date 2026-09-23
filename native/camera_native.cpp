// camera_native.cpp - fused C++ kernel for detect_ball_warp's own hot
// preamble (bot/vision.py): cv2.remap (fisheye unwarp) + cv2.cvtColor
// (BGR->HSV) + _sat_boost + cv2.inRange, currently four SEPARATE
// full-(n_r x n_theta)-frame passes each allocating and writing their
// own output array. Fused into one pass per output pixel - compute
// the bilinear-sampled BGR, convert to HSV, save the raw S, apply the
// saturation boost, and threshold, all before moving to the next pixel,
// instead of walking the whole image four times over four different
// arrays.
//
// NOT a reimplementation of what OpenCV is bad at - cv2's own C++ is
// already fast per-call. The win here is specifically kernel fusion
// (memory-bandwidth/cache locality from one pass instead of four), the
// same category of optimisation a single generic library call can't do
// automatically across an arbitrary multi-step pipeline. _find_ball_
// columns/_subpixel_centre (small-window operations downstream) are
// untouched - already cheap, not worth the risk of reimplementing.
//
// HSV conversion is a standard textbook BGR->HSV (H 0..179, S/V 0..255,
// matching cv2.cvtColor's own output range for 8-bit images), NOT
// verified bit-exact against OpenCV's own fixed-point/table-based
// implementation - empirically found to differ from cv2.cvtColor by at
// most +-1 per channel on random test images (rounding only, see
// native/test_camera_native.py), never more. Documented honestly rather
// than claimed as exact; a +-1 HSV difference is far inside this
// pipeline's own existing thresholds' tolerance (lower/upper bounds
// spanning dozens of units), so it doesn't change which pixels pass
// cv2.inRange in practice, but it's not literally the same code path.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return static_cast<uint8_t>(v);
}

// Bilinear-sample BGR at floating point (fx, fy) from an (H, W, 3) uint8
// image, matching cv2.remap's own INTER_LINEAR + BORDER_CONSTANT (0)
// behaviour for out-of-bounds samples.
static inline void sample_bgr(const uint8_t *frame, int H, int W,
                              double fx, double fy, double out[3]) {
    if (fx < 0 || fy < 0 || fx > W - 1 || fy > H - 1) {
        out[0] = out[1] = out[2] = 0.0;
        return;
    }
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = std::min(x0 + 1, W - 1);
    int y1 = std::min(y0 + 1, H - 1);
    double ax = fx - x0, ay = fy - y0;

    for (int c = 0; c < 3; ++c) {
        double v00 = frame[(y0 * W + x0) * 3 + c];
        double v10 = frame[(y0 * W + x1) * 3 + c];
        double v01 = frame[(y1 * W + x0) * 3 + c];
        double v11 = frame[(y1 * W + x1) * 3 + c];
        double top = v00 + (v10 - v00) * ax;
        double bot = v01 + (v11 - v01) * ax;
        out[c] = top + (bot - top) * ay;
    }
}

// Standard BGR->HSV (H 0..179, S/V 0..255) - see this file's own header
// comment for the +-1-vs-cv2.cvtColor caveat.
static inline void bgr_to_hsv(double b, double g, double r,
                              uint8_t &h_out, uint8_t &s_out, uint8_t &v_out) {
    double v = std::max({b, g, r});
    double vmin = std::min({b, g, r});
    double diff = v - vmin;
    double s = (v <= 0.0) ? 0.0 : (diff * 255.0 / v);
    double h;
    if (diff <= 1e-9) {
        h = 0.0;
    } else if (v == r) {
        h = 60.0 * (g - b) / diff;
    } else if (v == g) {
        h = 120.0 + 60.0 * (b - r) / diff;
    } else {
        h = 240.0 + 60.0 * (r - g) / diff;
    }
    if (h < 0.0) h += 360.0;
    int h180 = static_cast<int>(std::lround(h / 2.0)) % 180;
    if (h180 < 0) h180 += 180;
    h_out = static_cast<uint8_t>(h180);
    s_out = clamp_u8(static_cast<int>(std::lround(s)));
    v_out = clamp_u8(static_cast<int>(std::lround(v)));
}

static py::tuple unwarp_threshold(
        py::array_t<uint8_t> frame,
        py::array_t<float> map_x, py::array_t<float> map_y,
        int h_lo, int s_lo, int v_lo, int h_hi, int s_hi, int v_hi,
        bool sat_boost_enabled, double sat_boost_k, double sat_boost_mid) {
    auto f = frame.unchecked<3>();
    int H = static_cast<int>(f.shape(0));
    int W = static_cast<int>(f.shape(1));

    auto mx = map_x.unchecked<2>();
    auto my = map_y.unchecked<2>();
    py::ssize_t n_r = mx.shape(0);
    py::ssize_t n_theta = mx.shape(1);

    auto hsv_out = py::array_t<uint8_t>({n_r, n_theta, static_cast<py::ssize_t>(3)});
    auto s_raw_out = py::array_t<uint8_t>({n_r, n_theta});
    auto mask_out = py::array_t<uint8_t>({n_r, n_theta});
    auto hsv = hsv_out.mutable_unchecked<3>();
    auto s_raw = s_raw_out.mutable_unchecked<2>();
    auto mask = mask_out.mutable_unchecked<2>();

    // logistic saturation boost, matching _sat_boost's own constants exactly
    double lo = 1.0 / (1.0 + std::exp(sat_boost_k * sat_boost_mid));
    double hi = 1.0 / (1.0 + std::exp(-sat_boost_k * (1.0 - sat_boost_mid)));

    const uint8_t *frame_ptr = f.data(0, 0, 0);

    #pragma omp parallel for schedule(static)
    for (py::ssize_t i = 0; i < n_r; ++i) {
        for (py::ssize_t j = 0; j < n_theta; ++j) {
            double bgr[3];
            sample_bgr(frame_ptr, H, W, mx(i, j), my(i, j), bgr);
            // cv2.remap's own output is a uint8 image (rounded), and
            // cv2.cvtColor then runs on THAT rounded image, not on the
            // raw bilinear float - round here first to match, otherwise
            // low-V pixels (S = diff*255/V) amplify the skipped rounding
            // into large S errors. Confirmed directly: this was the
            // actual cause of the largest mismatches found while testing
            // (up to 7 S-units before this fix, <=1 after).
            double bgr_r[3] = {
                static_cast<double>(clamp_u8(static_cast<int>(std::lround(bgr[0])))),
                static_cast<double>(clamp_u8(static_cast<int>(std::lround(bgr[1])))),
                static_cast<double>(clamp_u8(static_cast<int>(std::lround(bgr[2])))),
            };
            uint8_t hh, ss, vv;
            bgr_to_hsv(bgr_r[0], bgr_r[1], bgr_r[2], hh, ss, vv);

            s_raw(i, j) = ss;

            uint8_t s_final = ss;
            if (sat_boost_enabled) {
                double sf = ss * (1.0 / 255.0);
                sf = 1.0 / (1.0 + std::exp(-sat_boost_k * (sf - sat_boost_mid)));
                sf = (sf - lo) * (255.0 / (hi - lo));
                s_final = clamp_u8(static_cast<int>(std::lround(sf)));
            }

            hsv(i, j, 0) = hh;
            hsv(i, j, 1) = s_final;
            hsv(i, j, 2) = vv;

            bool in_range = (hh >= h_lo && hh <= h_hi
                             && s_final >= s_lo && s_final <= s_hi
                             && vv >= v_lo && vv <= v_hi);
            mask(i, j) = in_range ? 255 : 0;
        }
    }

    return py::make_tuple(hsv_out, s_raw_out, mask_out);
}

PYBIND11_MODULE(camera_native, m) {
    m.doc() = "Fused unwarp + BGR->HSV + saturation boost + inRange threshold "
              "(bot/vision.py's detect_ball_warp own hot preamble).";
    m.def("unwarp_threshold", &unwarp_threshold,
          "(frame, map_x, map_y, h_lo, s_lo, v_lo, h_hi, s_hi, v_hi, "
          "sat_boost_enabled, sat_boost_k, sat_boost_mid) -> (hsv, s_raw, mask)");
}
