"""Tests for the detection pipeline.

Uses stdlib unittest so it runs with no extra dependencies:

    python -m unittest discover -s tests -v

Tests that need a trained model or the downloaded dataset skip themselves when
those are absent, so the suite is green on a fresh clone.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mmdetect.config import (  # noqa: E402
    IMAGE_MODEL_PATH,
    IMAGE_TRAIN_NPZ,
    LABEL_AI,
    LABEL_HUMAN,
    TEXT_MODEL_PATH,
)
from mmdetect.registry import MODALITIES, get_modality  # noqa: E402
from mmdetect.result import DetectionResult  # noqa: E402
from mmdetect.text.features import (  # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    StylometricFeatures,
    describe,
)


class TestLabelConvention(unittest.TestCase):
    """The single rule the whole project depends on: 1 = AI, 0 = human."""

    def test_constants(self):
        self.assertEqual(LABEL_AI, 1)
        self.assertEqual(LABEL_HUMAN, 0)

    def test_cifake_flip_is_applied(self):
        """CIFAKE ships 0=FAKE/1=REAL; the cached data must be flipped."""
        if not IMAGE_TRAIN_NPZ.exists():
            self.skipTest("image dataset not prepared")
        data = np.load(IMAGE_TRAIN_NPZ)
        labels = data["labels"]
        self.assertTrue(set(np.unique(labels)) <= {0, 1})
        # The prepared split is balanced, so a flip error would not show up as
        # a count difference - this asserts the values are at least in range
        # and both classes are present.
        self.assertGreater((labels == LABEL_AI).sum(), 0)
        self.assertGreater((labels == LABEL_HUMAN).sum(), 0)

    def test_flip_logic_directly(self):
        """The arithmetic used in image_data, checked in isolation."""
        cifake = np.array([0, 1, 0, 1])   # FAKE, REAL, FAKE, REAL
        ours = 1 - cifake
        np.testing.assert_array_equal(ours, [1, 0, 1, 0])  # AI, human, AI, human


class TestStylometricFeatures(unittest.TestCase):
    def test_shape_and_names(self):
        m = StylometricFeatures().fit_transform(["Hello world. Again!", "x"])
        self.assertEqual(m.shape, (2, N_FEATURES))
        self.assertEqual(len(FEATURE_NAMES), N_FEATURES)

    def test_empty_string_is_safe(self):
        m = StylometricFeatures().transform([""])
        self.assertEqual(m.shape, (1, N_FEATURES))
        self.assertTrue(np.all(np.isfinite(m)))

    def test_no_nan_or_inf_on_odd_input(self):
        odd = ["", "   ", "!!!", "1234567890", "a", "\n\n\n", "no terminator"]
        m = StylometricFeatures().transform(odd)
        self.assertTrue(np.all(np.isfinite(m)), "features must never be NaN/inf")

    def test_burstiness_discriminates(self):
        """Uniform sentence lengths should score lower burstiness than varied."""
        uniform = "I like cats a lot. I like dogs a lot. I like birds a lot."
        varied = ("I like cats. " + "Yesterday I walked through the old market "
                  "square where the vendors were packing away their stalls "
                  "under a darkening sky. " + "It rained.")
        self.assertLess(
            describe(uniform)["std_sent_len"],
            describe(varied)["std_sent_len"],
        )

    def test_describe_matches_feature_names(self):
        d = describe("A sentence. Another one!")
        self.assertEqual(list(d.keys()), FEATURE_NAMES)

    def test_no_absolute_length_features(self):
        """Raw size counts must stay out of the feature set.

        HC3's human answers are shorter than its AI answers, so any absolute
        length feature teaches the model "short means human" and it then
        mislabels short AI text. Guarding the names keeps that from creeping
        back in.
        """
        banned = {"n_chars", "n_words", "n_sentences", "n_paragraphs",
                  "avg_para_len"}
        self.assertEqual(banned & set(FEATURE_NAMES), set())

    def test_features_are_scale_invariant(self):
        """Repeating a document must barely move its style profile.

        The base passage is deliberately longer than ``VOCAB_WINDOW`` so the
        vocabulary window is already full; that is the regime the window is
        designed for and the one the data pipeline guarantees via MIN_CHARS.
        """
        from mmdetect.text.features import VOCAB_WINDOW

        base = (
            "The meeting ran long. Nobody had prepared, which was predictable "
            "given how the agenda went out barely an hour beforehand. Priya "
            "raised the budget question again and the room went quiet, because "
            "none of us had an answer that would survive contact with finance. "
            "We agreed to reconvene on Thursday! Then everyone left, slightly "
            "annoyed, to reschedule the afternoon they had just lost."
        )
        self.assertGreater(len(base.split()), VOCAB_WINDOW,
                           "test passage must exceed the vocabulary window")

        once = describe(base)
        thrice = describe(" ".join([base] * 3))
        for name in FEATURE_NAMES:
            a, b = once[name], thrice[name]
            scale = max(abs(a), abs(b), 1e-6)
            self.assertLess(
                abs(a - b) / scale, 0.30,
                f"feature '{name}' changed with document length: {a} -> {b}",
            )


class TestDetectionResult(unittest.TestCase):
    def test_ai_verdict(self):
        r = DetectionResult(0.92, "text")
        self.assertTrue(r.is_ai)
        self.assertEqual(r.verdict, "AI-generated")
        self.assertAlmostEqual(r.human_probability, 0.08, places=6)

    def test_human_verdict(self):
        r = DetectionResult(0.04, "image")
        self.assertFalse(r.is_ai)
        self.assertEqual(r.verdict, "Human / Real")

    def test_uncertain_band(self):
        """Scores hugging the threshold must not be reported as a decision."""
        for p in (0.45, 0.5, 0.55):
            self.assertEqual(DetectionResult(p, "text").verdict, "Uncertain")

    def test_confidence_scaling(self):
        self.assertAlmostEqual(DetectionResult(0.5, "text").confidence, 0.0)
        self.assertAlmostEqual(DetectionResult(1.0, "text").confidence, 1.0)
        self.assertAlmostEqual(DetectionResult(0.0, "text").confidence, 1.0)

    def test_as_dict_is_serialisable(self):
        import json

        d = DetectionResult(0.7, "text", "m").as_dict()
        json.loads(json.dumps(d))  # must not raise
        self.assertIn("verdict", d)


class TestRegistry(unittest.TestCase):
    def test_four_modalities(self):
        self.assertEqual(set(MODALITIES), {"text", "image", "audio", "video"})

    def test_weeks_are_ordered(self):
        weeks = [MODALITIES[k].week for k in ("text", "image", "audio", "video")]
        self.assertEqual(weeks, sorted(weeks))

    def test_ready_flags(self):
        self.assertTrue(get_modality("text").is_ready)
        self.assertTrue(get_modality("image").is_ready)
        self.assertFalse(get_modality("audio").is_ready)
        self.assertFalse(get_modality("video").is_ready)

    def test_planned_modality_raises_clearly(self):
        with self.assertRaises(NotImplementedError):
            get_modality("audio").detector()

    def test_unknown_modality(self):
        with self.assertRaises(KeyError):
            get_modality("hologram")


class TestTextDetector(unittest.TestCase):
    """End-to-end checks; skipped when the model has not been trained."""

    @classmethod
    def setUpClass(cls):
        if not TEXT_MODEL_PATH.exists():
            raise unittest.SkipTest("text model not trained")
        from mmdetect.text.infer import get_text_detector

        cls.det = get_text_detector()
        cls.det.load()

    def test_returns_valid_probability(self):
        r = self.det.predict("The quick brown fox jumps over the lazy dog. " * 12)
        self.assertGreaterEqual(r.ai_probability, 0.0)
        self.assertLessEqual(r.ai_probability, 1.0)
        self.assertEqual(r.modality, "text")

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            self.det.predict("   ")

    def test_short_text_is_flagged(self):
        r = self.det.predict("Too short to judge.")
        self.assertTrue(r.meta["too_short"])

    def test_explanation_present(self):
        r = self.det.predict("A reasonably long passage of writing. " * 20)
        self.assertIn("raw_features", r.explanation)
        self.assertIn("signals", r.explanation)

    def test_batch_matches_single(self):
        texts = ["First document about weather. " * 15,
                 "Second document about markets. " * 15]
        batch = self.det.predict_batch(texts)
        for t, b in zip(texts, batch):
            single = self.det.predict(t)
            self.assertAlmostEqual(single.ai_probability, b.ai_probability, places=6)


class TestImageDetector(unittest.TestCase):
    """End-to-end checks; skipped when the model has not been trained."""

    @classmethod
    def setUpClass(cls):
        if not IMAGE_MODEL_PATH.exists():
            raise unittest.SkipTest("image model not trained")
        from mmdetect.image.infer import get_image_detector

        cls.det = get_image_detector()
        cls.det.load()

    def _img(self, w=64, h=64):
        from PIL import Image

        rng = np.random.default_rng(0)
        return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))

    def test_returns_valid_probability(self):
        r = self.det.predict(self._img())
        self.assertGreaterEqual(r.ai_probability, 0.0)
        self.assertLessEqual(r.ai_probability, 1.0)
        self.assertEqual(r.modality, "image")

    def test_native_32x32_uses_global_view_only(self):
        r = self.det.predict(self._img(32, 32))
        self.assertTrue(r.explanation["native_resolution"])
        self.assertAlmostEqual(
            r.ai_probability, r.explanation["global_probability"], places=6
        )

    def test_large_image_is_tiled(self):
        r = self.det.predict(self._img(400, 300))
        self.assertGreater(r.explanation["n_tiles"], 1)
        self.assertTrue(r.meta["far_from_training_resolution"])

    def test_tiny_image_does_not_crash(self):
        r = self.det.predict(self._img(16, 16))
        self.assertGreaterEqual(r.ai_probability, 0.0)
        self.assertTrue(r.meta["upscaled_from_small"])

    def test_grayscale_is_converted(self):
        from PIL import Image

        gray = Image.fromarray(
            np.random.default_rng(1).integers(0, 255, (64, 64), dtype=np.uint8),
            mode="L",
        )
        r = self.det.predict(gray)
        self.assertGreaterEqual(r.ai_probability, 0.0)

    def test_score_batch_shape(self):
        rng = np.random.default_rng(2)
        batch = rng.integers(0, 255, (7, 32, 32, 3), dtype=np.uint8)
        probs = self.det.score_batch(batch)
        self.assertEqual(probs.shape, (7,))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
