"""Offline fitter for bot/vision.py's fish-eye ground-distance model. vision.py's comments
tell you to re-run the fit rather than rescale if crop_* changes; this is that fit.

It fits several model sizes and keeps the one with the lowest leave-one-out error instead of
a hand-picked degree.

Two details come from how the lens model is written:

1. The basis. vision.py models ground distance as an odd-power polynomial in t = tan(r / b),
   `ground = a*t + c*t^3 + d*t^5 + e*t^7 + f*t^9`, not a plain polynomial in r. So "degree" here is
   the number of terms in that basis (1..5), fitted by least squares. `coefficients_for_vision`
   prints the result in the form vision.py wants pasted.
2. The camera-geometry tag. A calibration is only valid for the frame resolution, crop rectangle
   and px scale it was fitted at, so all three are saved with it and checked on load.

Samples are a list of dicts with `radius_px` (pixel radius from the optical centre in
calibration-space pixels, i.e. after fisheye_px_scale, which is what _fisheye_poly takes) and
`ground_mm` (tape-measured ground distance to the object's contact point).

Run:  python fisheye_fit.py samples.json          (fit + print, no file written)
      python fisheye_fit.py samples.json --out cal.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np

DEFAULT_CALIBRATION_FILE = "fisheye_ground_calibration.json"
# vision.py's _fisheye_b (the lens constant the coefficients were fitted against)
DEFAULT_B = -510.76033
MAX_TERMS = 5 # a*t + c*t^3 + d*t^5 + e*t^7 + f*t^9


def _basis(t_values, terms):
    """odd-power design matrix, `terms` columns: t, t^3, t^5, ... (vision.py's basis)."""
    t = np.asarray(t_values, dtype=np.float64)
    return np.column_stack([t ** (2 * k + 1) for k in range(terms)])


def _fit_coefficients(radius_px, ground_mm, terms, b):
    """least-squares coefficients for `terms` odd powers of tan(r / b): the normal-equation solve this
    lens model needs (not a plain polynomial in r)."""
    r = np.asarray(radius_px, dtype=np.float64) / float(b)
    A = _basis(np.tan(r), terms)
    coef, *_ = np.linalg.lstsq(A, np.asarray(ground_mm, dtype=np.float64), rcond=None)
    return [float(v) for v in coef]


def _predict(coeffs, radius_px, b):
    coeffs = list(coeffs)
    row = _basis([math.tan(float(radius_px) / float(b))], len(coeffs))[0]
    return float(np.dot(row, np.asarray(coeffs, dtype=np.float64)))


def _calculate_rmse(actual_values, predicted_values):
    """root-mean-square error between two equal-length lists."""
    if not actual_values:
        return 0.0
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values)]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def _leave_one_out_rmse(samples, terms, b):
    """refit with each sample held out and measure the error on the held-out one.
    """
    if len(samples) < terms + 2:
        return None

    predicted_values = []
    actual_values = []
    for sample_index, held_out_sample in enumerate(samples):
        training_samples = samples[:sample_index] + samples[sample_index + 1 :]
        coefficients = _fit_coefficients(
            [s["radius_px"] for s in training_samples],
            [s["ground_mm"] for s in training_samples],
            terms,
            b,
        )
        predicted_values.append(_predict(coefficients, held_out_sample["radius_px"], b))
        actual_values.append(held_out_sample["ground_mm"])
    return _calculate_rmse(actual_values, predicted_values)


def fit_ground_calibration(samples, max_terms=MAX_TERMS, b=DEFAULT_B):
    """Fit several model sizes and choose the best one by validation error.

    Candidates are every term count from 1 to
    max_terms, each gets a training RMSE and (where there are enough samples) a leave-one-out RMSE, and
    the winner is the lowest leave-one-out error, falling back to training error when no candidate could
    be validated. The extra terms are there so the validation can reject them.
    """
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to fit a ground-distance model.")
    for s in samples:
        if "radius_px" not in s or "ground_mm" not in s:
            raise ValueError("every sample needs radius_px and ground_mm")
    if b == 0:
        raise ValueError("b must be non-zero (tan(r / b))")

    max_terms = max(1, min(int(max_terms), MAX_TERMS, len(samples) - 1))
    radius_values = [float(s["radius_px"]) for s in samples]
    ground_values = [float(s["ground_mm"]) for s in samples]

    candidate_models = []
    for terms in range(1, max_terms + 1):
        coefficients = _fit_coefficients(radius_values, ground_values, terms, b)
        training_predictions = [_predict(coefficients, r, b) for r in radius_values]
        candidate_models.append(
            {
                "terms": terms,
                "exponents": [2 * k + 1 for k in range(terms)],
                "coefficients": coefficients,
                "training_rmse_mm": _calculate_rmse(ground_values, training_predictions),
                "leave_one_out_rmse_mm": _leave_one_out_rmse(samples, terms, b),
            }
        )

    comparable_models = [m for m in candidate_models if m["leave_one_out_rmse_mm"] is not None]
    if comparable_models:
        selected_model = min(comparable_models, key=lambda m: m["leave_one_out_rmse_mm"])
        selection_reason = "lowest leave-one-out RMSE"
    else:
        selected_model = min(candidate_models, key=lambda m: m["training_rmse_mm"])
        selection_reason = "lowest training RMSE"

    return {
        "model_type": "odd_power_tan",
        "basis": "sum(a_k * tan(r / b)^(2k+1))",
        "b": float(b),
        "selected_terms": selected_model["terms"],
        "exponents": selected_model["exponents"],
        "coefficients": selected_model["coefficients"],
        "selection_reason": selection_reason,
        "fit_metrics": candidate_models,
        "sample_count": len(samples),
        "radius_px_range": [min(radius_values), max(radius_values)],
        "ground_mm_range": [min(ground_values), max(ground_values)],
    }


def vision_constants(model):
    """The fitted model as the two lines to paste into bot/vision.py, plus the worst-case error."""
    coeffs = list(model["coefficients"])
    while len(coeffs) < MAX_TERMS:
        coeffs.append(0.0)
    return (
        f"_fisheye_b = {model['b']:.5f}\n"
        f"_fisheye_c = ({coeffs[0]:.5f}, {coeffs[1]:.5f}, {coeffs[2]:.5f}, "
        f"{coeffs[3]:.5f}, {coeffs[4]:.5f})"
    )


def resolve_calibration_path(calibration_file=DEFAULT_CALIBRATION_FILE):
    """Resolve relative calibration paths against this script's directory.
    """
    if os.path.isabs(calibration_file):
        return calibration_file
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), calibration_file)


def save_calibration(samples, geometry, max_terms=MAX_TERMS, b=DEFAULT_B,
                     calibration_file=DEFAULT_CALIBRATION_FILE):
    """Fit a model from samples and save it to disk as JSON (atomic write).

    geometry is the camera configuration the samples were taken under ({width, height, crop,
    px_scale}), which is what makes a stale calibration detectable later (see load_calibration).
    """
    model = fit_ground_calibration(samples, max_terms=max_terms, b=b)
    calibration_data = {
        "version": 1,
        "camera_geometry": {
            "width": int(geometry["width"]),
            "height": int(geometry["height"]),
            "crop": list(geometry.get("crop") or ()),
            "fisheye_px_scale": float(geometry.get("fisheye_px_scale", 1.0)),
        },
        "input_feature": "radius_px_from_optical_centre (calibration-space, post px_scale)",
        "distance_units": "mm",
        "model": model,
        "samples": [
            {"radius_px": float(s["radius_px"]), "ground_mm": float(s["ground_mm"])}
            for s in samples
        ],
    }

    calibration_path = resolve_calibration_path(calibration_file)
    temp_path = f"{calibration_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as calibration_handle:
        json.dump(calibration_data, calibration_handle, indent=2)
        calibration_handle.flush()
        os.fsync(calibration_handle.fileno())
    os.replace(temp_path, calibration_path)
    return calibration_data, calibration_path


def _geometry_matches(saved, current):
    if not isinstance(saved, dict) or not saved:
        return False
    try:
        if (int(saved["width"]), int(saved["height"])) != (int(current["width"]), int(current["height"])):
            return False
        if [float(v) for v in (saved.get("crop") or ())] != [float(v) for v in (current.get("crop") or ())]:
            return False
        return abs(float(saved.get("fisheye_px_scale", 1.0))
                   - float(current.get("fisheye_px_scale", 1.0))) < 1e-9
    except (KeyError, TypeError, ValueError):
        return False


def load_calibration(geometry, calibration_file=DEFAULT_CALIBRATION_FILE):
    """Load a saved calibration if it matches the current camera geometry (resolution, crop and px
    scale), else None."""
    calibration_path = resolve_calibration_path(calibration_file)
    if not os.path.isfile(calibration_path):
        return None

    try:
        with open(calibration_path, encoding="utf-8") as calibration_handle:
            calibration_data = json.load(calibration_handle)
    except (OSError, json.JSONDecodeError):
        return None

    if not _geometry_matches(calibration_data.get("camera_geometry"), geometry):
        return None

    model = calibration_data.get("model")
    if not isinstance(model, dict):
        return None
    if model.get("model_type") != "odd_power_tan":
        return None
    coefficients = model.get("coefficients")
    if not isinstance(coefficients, list) or not coefficients:
        return None
    return calibration_data


def predict_ground_mm(calibration_data, radius_px):
    """Predict ground distance in mm from a pixel radius, clamped to the fitted range (outside it
    the polynomial is extrapolating)."""
    if calibration_data is None or radius_px is None:
        return None
    model = calibration_data.get("model", {})
    coefficients = model.get("coefficients")
    radius_range = model.get("radius_px_range")
    if not isinstance(coefficients, list) or not coefficients:
        return None
    if not isinstance(radius_range, list) or len(radius_range) != 2:
        return None

    clamped_radius = min(max(float(radius_px), float(radius_range[0])), float(radius_range[1]))
    predicted = _predict(coefficients, clamped_radius, model.get("b", DEFAULT_B))
    if predicted < 0:
        return None
    return predicted


def _load_samples(path):
    with open(path, encoding="utf-8") as handle:
        if path.lower().endswith(".csv"):
            samples = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                radius, ground = (float(v) for v in line.replace(",", " ").split()[:2])
                samples.append({"radius_px": radius, "ground_mm": ground})
            return samples
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return data["samples"]
    raise ValueError("samples file must be a JSON list, {'samples': [...]}, or a two-column CSV")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit bot/vision.py's fish-eye ground-distance model from bench samples.")
    parser.add_argument("samples", help="JSON or CSV of {radius_px, ground_mm} samples")
    parser.add_argument("--max-terms", type=int, default=MAX_TERMS,
                        help=f"largest model to consider, 1..{MAX_TERMS} (default {MAX_TERMS})")
    parser.add_argument("--b", type=float, default=DEFAULT_B,
                        help="the lens constant tan(r / b); defaults to vision.py's _fisheye_b")
    parser.add_argument("--out", metavar="PATH", default=None, help="write the calibration JSON here")
    parser.add_argument("--geometry", default=None,
                        help="JSON camera geometry to tag the calibration with, e.g. "
                             '\'{"width":640,"height":480,"crop":[0,0,0,0],"fisheye_px_scale":1.0}\'')
    args = parser.parse_args(argv)

    samples = _load_samples(args.samples)
    model = fit_ground_calibration(samples, max_terms=args.max_terms, b=args.b)

    print(f"{model['sample_count']} samples, "
          f"radius {model['radius_px_range'][0]:.1f}-{model['radius_px_range'][1]:.1f} px, "
          f"ground {model['ground_mm_range'][0]:.0f}-{model['ground_mm_range'][1]:.0f} mm")
    print(f"{'terms':>5} {'train RMSE':>12} {'leave-one-out':>14}")
    for candidate in model["fit_metrics"]:
        loo = candidate["leave_one_out_rmse_mm"]
        loo_text = "n/a" if loo is None else format(loo, ".2f")
        print(f"{candidate['terms']:>5} {candidate['training_rmse_mm']:>12.2f} {loo_text:>14}")
    print(f"\nselected {model['selected_terms']} terms by {model['selection_reason']}")
    print("\npaste into bot/vision.py:")
    print(vision_constants(model))

    if args.out:
        geometry = json.loads(args.geometry) if args.geometry else {"width": 0, "height": 0, "crop": []}
        _, path = save_calibration(samples, geometry, max_terms=args.max_terms, b=args.b,
                                   calibration_file=args.out)
        print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
