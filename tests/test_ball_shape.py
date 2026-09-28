"""Tests for bot/vision.py's ball shape gate (_biggest_blob): a flat marking or stripe in the
ball's colour, like a red marker on another robot, mustn't beat a smaller round ball on pixel
count.
"""

import cv2
import numpy as np
import pytest

import bot.vision as vision


def _mask(h=200, w=300):
    return np.zeros((h, w), dtype=np.uint8)


class TestBiggestBlob:
    def test_no_blobs_returns_none(self):
        assert vision._biggest_blob(_mask()) is None

    def test_finds_a_round_blob(self):
        m = _mask()
        cv2.circle(m, (150, 100), 20, 255, -1)
        blob = vision._biggest_blob(m)
        assert blob is not None
        cx, cy, w, h, area = blob
        assert abs(cx - 150) < 2 and abs(cy - 100) < 2
        assert 35 <= w <= 42 and 35 <= h <= 42
        assert area > 1000

    def test_a_bigger_stripe_does_not_beat_a_smaller_ball(self):
        """a wide flat marking has far more pixels than the ball but isn't round; the ball wins"""
        stripe_area = 280 * 20 # the rectangle below
        ball_area = np.pi * 15 ** 2 # the circle below
        assert stripe_area > ball_area * 4 # the stripe really is the bigger blob; sanity check

        m = _mask()
        cv2.rectangle(m, (10, 90), (290, 110), 255, -1) # a long horizontal stripe
        cv2.circle(m, (150, 40), 15, 255, -1) # a small round ball

        blob = vision._biggest_blob(m)
        assert blob is not None
        cx, cy, w, h, area = blob
        assert abs(cy - 40) < 3 # the round ball, not the stripe at y=90-110
        assert area < 1000

    def test_a_lone_stripe_is_rejected_outright(self):
        """with no round blob in view, the stripe still isn't taken as the ball: no ball this
        frame beats locking onto a marker
        """
        m = _mask()
        cv2.rectangle(m, (10, 90), (290, 110), 255, -1)
        assert vision._biggest_blob(m) is None

    def test_a_square_marker_is_more_permissive_than_a_stripe(self):
        """a compact square fills about 0.64 of its enclosing circle, above the 0.45 threshold,
        so it passes. The gate catches long markings, not every false positive.
        """
        m = _mask()
        cv2.rectangle(m, (140, 30), (170, 60), 255, -1) # 30x30 square
        assert vision._biggest_blob(m) is not None

    def test_below_min_orange_px_is_ignored(self, monkeypatch):
        monkeypatch.setattr(vision, "min_orange_px", 50)
        m = _mask()
        cv2.circle(m, (150, 100), 3, 255, -1) # tiny, but round
        assert vision._biggest_blob(m) is None

    def test_fill_ratio_is_configurable(self, monkeypatch):
        m = _mask()
        cv2.rectangle(m, (10, 90), (290, 110), 255, -1)
        assert vision._biggest_blob(m) is None
        monkeypatch.setattr(vision, "ball_min_fill_ratio", 0.0)
        assert vision._biggest_blob(m) is not None


class TestBallCandidates:
    def test_reports_every_blob_with_its_fill_ratio(self):
        m = _mask()
        cv2.rectangle(m, (10, 90), (290, 110), 255, -1) # a stripe, low fill ratio
        cv2.circle(m, (150, 40), 15, 255, -1) # a ball, high fill ratio
        candidates = vision._ball_candidates(m)
        assert len(candidates) == 2
        ratios = sorted(cand[7] for cand in candidates)
        assert ratios[0] < vision.ball_min_fill_ratio <= ratios[1]

    def test_ignores_specks_below_min_orange_px(self, monkeypatch):
        monkeypatch.setattr(vision, "min_orange_px", 50)
        m = _mask()
        cv2.circle(m, (150, 100), 3, 255, -1)
        assert vision._ball_candidates(m) == []


class TestSuggestFillRatioThreshold:
    def test_no_samples_of_one_class_reports_nothing_to_separate(self):
        t, correct, total = vision.suggest_fill_ratio_threshold([])
        assert t is None and correct == 0 and total == 0
        t, correct, total = vision.suggest_fill_ratio_threshold(
            [(True, 0.9), (True, 0.8)])
        assert t is None and total == 2

    def test_clean_separation(self):
        samples = [(True, 0.9), (True, 0.85), (True, 0.7),
                  (False, 0.4), (False, 0.3), (False, 0.5)]
        t, correct, total = vision.suggest_fill_ratio_threshold(samples)
        assert t is not None
        assert correct == total == 6
        assert 0.5 < t <= 0.7

    def test_picks_the_widest_margin_among_perfect_cuts(self):
        """0.6 and 0.65 both split these perfectly; 0.6 is in the middle of the 0.5 to 0.7 gap
        and wins
        """
        samples = [(True, 0.7), (True, 0.8), (False, 0.5), (False, 0.4)]
        t, correct, total = vision.suggest_fill_ratio_threshold(samples)
        assert correct == total == 4
        assert t == pytest.approx(0.6, abs=1e-6)

    def test_overlap_is_reported_as_imperfect(self):
        """a decoy rounder than the smallest ball sample: no cut gets every sample right, and
        the count says so
        """
        samples = [(True, 0.9), (True, 0.6), (False, 0.65), (False, 0.3)]
        t, correct, total = vision.suggest_fill_ratio_threshold(samples)
        assert t is not None
        assert correct < total

    def test_matches_the_fixtures_used_to_test_biggest_blob(self):
        """TestBiggestBlob's stripe and ball, run through the threshold suggestion"""
        m = _mask()
        cv2.rectangle(m, (10, 90), (290, 110), 255, -1)
        cv2.circle(m, (150, 40), 15, 255, -1)
        candidates = vision._ball_candidates(m)
        by_area = sorted(candidates, key=lambda c: c[6])
        stripe_ratio, ball_ratio = by_area[1][7], by_area[0][7] # stripe: bigger area
        t, correct, total = vision.suggest_fill_ratio_threshold(
            [(True, ball_ratio), (False, stripe_ratio)])
        assert correct == total == 2
        assert stripe_ratio < t <= ball_ratio


class TestUnwrapScaling:
    """In the unwrap a far ball is a couple of columns wide and six rows tall, so the fill ratio
    is measured in ball-sized units. Unscaled, every ball past about a metre would fail.
    """

    @pytest.fixture(autouse=True)
    def _geometry(self):
        import bot.robot_select as robot_select
        robot_select.select_and_publish_config()
        g = vision._ring_geometry()
        self.r_lut, self.n_theta = g[2], len(g[3])

    def _ellipse_at(self, row, stretch=1.0):
        w, h = vision._ball_cells_at(row, self.r_lut, self.n_theta)
        m = np.zeros((len(self.r_lut), self.n_theta), dtype=np.uint8)
        axes = (max(1, int(round(stretch * w / 2))), max(1, int(round(h / 2))))
        cv2.ellipse(m, (self.n_theta // 2, row), axes, 0, 0, 360, 255, -1)
        return m

    @pytest.mark.parametrize("row", [60, 120, 200, 280, 400])
    def test_a_whole_ball_passes_at_every_range(self, row):
        blob = vision._biggest_blob(self._ellipse_at(row), self.r_lut, self.n_theta)
        assert blob is not None

    def test_the_unscaled_test_would_reject_a_far_ball(self):
        """the same far ball measured in raw cells fails"""
        m = self._ellipse_at(400)
        assert vision._biggest_blob(m) is None
        assert vision._biggest_blob(m, self.r_lut, self.n_theta) is not None

    @pytest.mark.parametrize("row", [60, 120, 200, 280, 400])
    def test_a_three_ball_wide_stripe_fails_at_every_range(self, row):
        m = self._ellipse_at(row, stretch=0.0)
        w, h = vision._ball_cells_at(row, self.r_lut, self.n_theta)
        c = self.n_theta // 2
        cv2.rectangle(m, (c - int(1.5 * w), row - int(h / 2)), (c + int(1.5 * w), row + int(h / 2)),
                      255, -1)
        assert vision._biggest_blob(m, self.r_lut, self.n_theta) is None

    def test_tiny_far_blobs_count_their_pixels(self):
        """a 2 x 6 blob counts as 12 pixels, not the near-zero area of the polygon through its
        pixel centres
        """
        m = np.zeros((len(self.r_lut), self.n_theta), dtype=np.uint8)
        m[398:404, 270:272] = 255
        (cand,) = vision._ball_candidates(m, self.r_lut, self.n_theta)
        assert cand[6] == 12
