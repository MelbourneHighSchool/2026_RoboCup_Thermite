"""Tests for bot/vision.py's _ring_geometry(), the unwrap the camera thread and the bench
tools share.
"""

import numpy as np
import pytest

import bot.robot_select as robot_select
import bot.vision as vision


@pytest.fixture(autouse=True)
def _per_bot_config():
    """_ring_geometry() reads the crop settings robot_select publishes"""
    robot_select.select_and_publish_config()


class TestRingGeometry:
    def test_shapes_agree_with_build_warp_maps(self):
        (map_x, map_y, r_lut, theta_lut, col_bearing, inner_blank, notch_region,
         inner_r, outer_r, ring_r) = vision._ring_geometry()
        assert map_x.shape == map_y.shape == (len(r_lut), len(theta_lut))
        assert len(theta_lut) == vision.warp_ntheta
        assert col_bearing.shape == theta_lut.shape
        assert 0 < ring_r <= inner_r < outer_r

    def test_notch_masks_only_the_wedge_about_the_dwibbler(self):
        _mx, _my, r_lut, _th, col_bearing, inner_blank, notch_region, inner_r, *_ = (
            vision._ring_geometry())
        if not vision.mouth_notch_enabled:
            pytest.skip("this bot has the mouth notch disabled")
        assert inner_blank is not None and notch_region is not None
        assert inner_blank.shape == notch_region.shape == (len(r_lut), len(col_bearing))
        # the two masks split the near rows between them: never both, and together all of them
        near = r_lut < inner_r
        assert not (inner_blank & notch_region).any()
        assert (inner_blank | notch_region)[near].all()
        assert not inner_blank[~near].any() and not notch_region[~near].any()
        # the wedge is centred on the dwibbler (bearing 0)
        wedge_cols = notch_region[near].any(axis=0)
        assert wedge_cols[np.abs(col_bearing) <= vision.mouth_notch_half_deg].all()
        assert not wedge_cols[np.abs(col_bearing) > vision.mouth_notch_half_deg].any()

    def test_matches_the_camera_threads_own_setup(self):
        """the same settings give exactly the maps the camera thread used to build itself"""
        crop_w = vision.sensor_size[0] - vision.crop_left - vision.crop_right
        crop_h = vision.sensor_size[1] - vision.crop_top - vision.crop_bottom
        ccx, ccy = crop_w // 2, crop_h // 2
        max_r = min(crop_w, crop_h) // 2
        inner_r = int(vision.exclusion_inner_frac * max_r)
        outer_r = int(vision.exclusion_outer_frac * max_r)
        notch_px = (int(round(vision.mouth_notch_px)) if vision.mouth_notch_enabled else 0)
        ring_r = max(1, inner_r - notch_px)
        map_x, map_y, r_lut, theta_lut = vision.build_warp_maps(
            ccx, ccy, ring_r, outer_r, None, vision.warp_ntheta)

        g = vision._ring_geometry()
        assert np.array_equal(g[0], map_x) and np.array_equal(g[1], map_y)
        assert np.array_equal(g[2], r_lut) and np.array_equal(g[3], theta_lut)
        assert g[7] == inner_r and g[8] == outer_r and g[9] == ring_r
