"""Unit tests for nepsyc/competence.py: the Nepali language competence probe.

No live API calls anywhere in this file -- scoring runs against sacrebleu directly
(a local library, not a network call), and the end-to-end test uses providers.MockProvider
via run_competence_sweep(cfg, mock=True), the same mock backend the main pipeline's
own smoke test (`python run.py evaluate --mock`) uses.

Run with:  python -m unittest tests.test_competence -v
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from nepsyc import competence
from nepsyc.config import CompetenceCfg, load_config


class TestBleuTokenizeFor(unittest.TestCase):
    def test_en_to_ne_devanagari_uses_intl(self):
        self.assertEqual(competence._bleu_tokenize_for("en_to_ne", "devanagari"), "intl")

    def test_en_to_ne_romanized_uses_13a(self):
        self.assertEqual(competence._bleu_tokenize_for("en_to_ne", "romanized"), "13a")

    def test_ne_to_en_always_13a_regardless_of_script(self):
        # The reference side of ne_to_en is English, so the source script must not matter.
        self.assertEqual(competence._bleu_tokenize_for("ne_to_en", "devanagari"), "13a")
        self.assertEqual(competence._bleu_tokenize_for("ne_to_en", "romanized"), "13a")

    def test_comprehension_follows_script(self):
        self.assertEqual(competence._bleu_tokenize_for("comprehension", "devanagari"), "intl")
        self.assertEqual(competence._bleu_tokenize_for("comprehension", "romanized"), "13a")


class TestVerdictFor(unittest.TestCase):
    def setUp(self):
        self.cfg = CompetenceCfg(
            understands_chrfpp_min=50.0, understands_bleu_min=25.0, partial_chrfpp_min=30.0,
        )

    def test_understands_requires_both_metrics_above_their_minimum(self):
        self.assertEqual(competence.verdict_for(30.0, 60.0, self.cfg), "Understands")

    def test_high_chrfpp_alone_is_only_partial(self):
        # chrF++ clears the Understands bar but BLEU does not -> Partial, not Understands.
        self.assertEqual(competence.verdict_for(10.0, 60.0, self.cfg), "Partial")

    def test_mid_chrfpp_is_partial(self):
        self.assertEqual(competence.verdict_for(0.0, 35.0, self.cfg), "Partial")

    def test_low_chrfpp_is_poor_even_with_high_bleu(self):
        self.assertEqual(competence.verdict_for(90.0, 10.0, self.cfg), "Poor")

    def test_boundary_values_are_inclusive(self):
        self.assertEqual(competence.verdict_for(25.0, 50.0, self.cfg), "Understands")
        self.assertEqual(competence.verdict_for(0.0, 30.0, self.cfg), "Partial")

    def test_missing_scores_yield_blank_verdict(self):
        self.assertEqual(competence.verdict_for(None, 60.0, self.cfg), "")
        self.assertEqual(competence.verdict_for(30.0, None, self.cfg), "")


class TestScoreReply(unittest.TestCase):
    def test_identical_reply_scores_near_maximum(self):
        scores = competence.score_reply("The sun rises in the east.",
                                        ["The sun rises in the east."], "ne_to_en", "devanagari")
        self.assertGreater(scores["bleu"]["value"], 99.0)
        self.assertGreater(scores["chrf++"]["value"], 99.0)

    def test_unrelated_reply_scores_low(self):
        scores = competence.score_reply("Completely unrelated sentence about cars.",
                                        ["The sun rises in the east."], "ne_to_en", "devanagari")
        self.assertLess(scores["bleu"]["value"], 20.0)
        self.assertLess(scores["chrf++"]["value"], 40.0)

    def test_empty_reply_does_not_crash_and_scores_zero(self):
        scores = competence.score_reply("", ["The sun rises in the east."], "ne_to_en", "devanagari")
        self.assertEqual(scores["bleu"]["value"], 0.0)
        self.assertEqual(scores["chrf++"]["value"], 0.0)

    def test_multiple_references_credits_the_closer_one(self):
        # "I like tea." should score much higher against a reference set that includes
        # it verbatim than against one that only has an unrelated sentence.
        with_match = competence.score_reply("I like tea.", ["I like coffee.", "I like tea."],
                                            "ne_to_en", "romanized")
        without_match = competence.score_reply("I like tea.", ["The weather is good today."],
                                                "ne_to_en", "romanized")
        self.assertGreater(with_match["chrf++"]["value"], without_match["chrf++"]["value"])

    def test_signature_reflects_requested_tokenizer(self):
        scores = competence.score_reply("test", ["test"], "en_to_ne", "devanagari")
        self.assertIn("tok:intl", scores["bleu"]["signature"])
        scores_rom = competence.score_reply("test", ["test"], "en_to_ne", "romanized")
        self.assertIn("tok:13a", scores_rom["bleu"]["signature"])


class TestLoadProbeSet(unittest.TestCase):
    def test_real_probe_set_loads_and_validates(self):
        rows = competence.load_probe_set("data/seeds/competence_probes.csv")
        self.assertEqual(len(rows), 24)
        for r in rows:
            self.assertIn(r["direction"], competence.DIRECTIONS)
            self.assertIn(r["script"], competence.SCRIPTS)
            self.assertIsInstance(r["reference_texts"], list)
            self.assertTrue(r["reference_texts"])

    def test_rejects_unknown_direction(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text(
                "seed_id,direction,script,source_text,reference_texts,notes\n"
                "X1,sideways,devanagari,hello,hi,\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                competence.load_probe_set(p)

    def test_rejects_blank_reference_texts(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.csv"
            p.write_text(
                "seed_id,direction,script,source_text,reference_texts,notes\n"
                "X1,en_to_ne,devanagari,hello,,\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                competence.load_probe_set(p)


class TestScoresRoundTrip(unittest.TestCase):
    def test_write_then_load_is_lossless(self):
        records = [
            {"model": "Test-Model", "seed_id": "CMP001", "direction": "en_to_ne",
             "script": "devanagari", "source_text": "The sun rises in the east.",
             "reference_texts": ["ref one", "ref two"], "reply": "some reply",
             "error": "", "metric": "bleu", "value": 12.3456, "signature": "sig-a"},
            {"model": "Test-Model", "seed_id": "CMP001", "direction": "en_to_ne",
             "script": "devanagari", "source_text": "The sun rises in the east.",
             "reference_texts": ["ref one", "ref two"], "reply": "some reply",
             "error": "", "metric": "chrf++", "value": 45.6, "signature": "sig-b"},
            {"model": "Test-Model", "seed_id": "CMP002", "direction": "ne_to_en",
             "script": "romanized", "source_text": "Malai chiya man parcha.",
             "reference_texts": ["I like tea."], "reply": "", "error": "timeout",
             "metric": "bleu", "value": float("nan"), "signature": ""},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "competence_scores.csv"
            competence.write_scores(records, path)
            loaded = competence.load_scores(path)

        self.assertEqual(len(loaded), len(records))
        for original, back in zip(records, loaded):
            self.assertEqual(back["model"], original["model"])
            self.assertEqual(back["seed_id"], original["seed_id"])
            self.assertEqual(back["reference_texts"], original["reference_texts"])
            # tables.read_csv's own convention: a blank cell comes back as None, never
            # "" (see tables.py's docstring) -- true of every CSV in this repo, not
            # just this one, so a blank original signature is expected to load as None.
            self.assertEqual(back["signature"], original["signature"] or None)
            if math.isnan(original["value"]):
                self.assertTrue(math.isnan(back["value"]))
            else:
                self.assertAlmostEqual(back["value"], original["value"], places=4)


class TestAggregate(unittest.TestCase):
    def test_per_direction_means_and_overall_verdict(self):
        cfg = CompetenceCfg(understands_chrfpp_min=50.0, understands_bleu_min=25.0, partial_chrfpp_min=30.0)
        records = [
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "bleu", "value": 40.0},
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "chrf++", "value": 60.0},
            {"model": "M", "seed_id": "B", "direction": "ne_to_en", "metric": "bleu", "value": 10.0},
            {"model": "M", "seed_id": "B", "direction": "ne_to_en", "metric": "chrf++", "value": 20.0},
        ]
        rows = {r["direction"]: r for r in competence.aggregate(records, cfg)}

        self.assertAlmostEqual(rows["en_to_ne"]["bleu"], 40.0)
        self.assertAlmostEqual(rows["en_to_ne"]["chrfpp"], 60.0)
        self.assertEqual(rows["en_to_ne"]["verdict"], "")  # only "overall" carries a verdict
        self.assertEqual(rows["en_to_ne"]["n_items"], 1)

        # overall pools both seeds: bleu mean (40+10)/2=25.0, chrf++ mean (60+20)/2=40.0.
        # chrf++ 40 falls short of understands_chrfpp_min (50) but clears partial_chrfpp_min
        # (30), so the combined result is "Partial" even though en_to_ne alone would have
        # been "Understands" -- pooling directions can pull the overall verdict down.
        self.assertAlmostEqual(rows["overall"]["bleu"], 25.0)
        self.assertAlmostEqual(rows["overall"]["chrfpp"], 40.0)
        self.assertEqual(rows["overall"]["n_items"], 2)
        self.assertEqual(rows["overall"]["verdict"], "Partial")

    def test_direction_with_no_data_is_still_reported_with_none_scores(self):
        cfg = CompetenceCfg()
        records = [
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "bleu", "value": 10.0},
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "chrf++", "value": 10.0},
        ]
        rows = {r["direction"]: r for r in competence.aggregate(records, cfg)}
        self.assertIsNone(rows["comprehension"]["bleu"])
        self.assertIsNone(rows["comprehension"]["chrfpp"])
        self.assertEqual(rows["comprehension"]["n_items"], 0)

    def test_errored_probe_rows_are_excluded_from_the_mean(self):
        cfg = CompetenceCfg()
        records = [
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "bleu", "value": 80.0},
            {"model": "M", "seed_id": "A", "direction": "en_to_ne", "metric": "chrf++", "value": 80.0},
            {"model": "M", "seed_id": "B", "direction": "en_to_ne", "metric": "bleu", "value": float("nan")},
            {"model": "M", "seed_id": "B", "direction": "en_to_ne", "metric": "chrf++", "value": float("nan")},
        ]
        rows = {r["direction"]: r for r in competence.aggregate(records, cfg)}
        self.assertAlmostEqual(rows["en_to_ne"]["bleu"], 80.0)
        self.assertEqual(rows["en_to_ne"]["n_items"], 1)


class TestEndToEndMockSweep(unittest.TestCase):
    """Mirrors how `python run.py evaluate --mock` smoke-tests the main pipeline:
    no network, deterministic, proves the sweep writes valid CSV that round-trips."""

    def test_mock_sweep_writes_valid_lossless_csv(self):
        cfg = load_config()
        cfg.run.target_model_ids = ["Llama-3.1-8B"]
        with tempfile.TemporaryDirectory() as td:
            cfg.competence.output_dir = td
            result = competence.run_competence_sweep(cfg, mock=True)

            probes = competence.load_probe_set(cfg.competence.probe_set)
            expected_rows = len(probes) * len(competence.METRICS)  # one target model
            self.assertEqual(len(result["records"]), expected_rows)

            scores_path = Path(result["paths"]["competence_scores"])
            summary_path = Path(result["paths"]["competence_summary"])
            self.assertTrue(scores_path.exists())
            self.assertTrue(summary_path.exists())

            reloaded = competence.load_scores(scores_path)
            self.assertEqual(len(reloaded), expected_rows)
            for original, back in zip(result["records"], reloaded):
                self.assertEqual(back["seed_id"], original["seed_id"])
                self.assertEqual(back["reference_texts"], original["reference_texts"])

            # 4 rows per model (3 directions + overall); exactly one model was selected.
            self.assertEqual(len(result["summary"]), 4)
            overall = next(r for r in result["summary"] if r["direction"] == "overall")
            self.assertIn(overall["verdict"], ("Understands", "Partial", "Poor"))

    def test_sweep_does_not_touch_sycophancy_pipeline_files(self):
        cfg = load_config()
        cfg.run.target_model_ids = ["Llama-3.1-8B"]
        with tempfile.TemporaryDirectory() as td:
            cfg.competence.output_dir = td
            competence.run_competence_sweep(cfg, mock=True)
            written = {p.name for p in Path(td).iterdir()}
        self.assertEqual(written, {"competence_scores.csv", "competence_summary.csv"})


if __name__ == "__main__":
    unittest.main()
