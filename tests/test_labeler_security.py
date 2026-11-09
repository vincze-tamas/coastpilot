import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


LABELER_PATH = Path(__file__).resolve().parents[1] / "tools" / "labeler_app.py"


class TestLabelerStartupSecurity(unittest.TestCase):
    def run_labeler(self, env):
        with patch.dict(os.environ, env, clear=True), patch("flask.Flask.run") as run:
            runpy.run_path(str(LABELER_PATH), run_name="__main__")
            return run.call_args.kwargs

    def test_default_bind_is_loopback(self):
        self.assertEqual(self.run_labeler({})["host"], "127.0.0.1")

    def test_explicit_loopback_bind_allows_no_token(self):
        for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
            with self.subTest(host=host):
                self.assertEqual(self.run_labeler({"COASTPILOT_LABEL_HOST": host})["host"], host)

    def test_non_loopback_bind_without_real_token_never_starts(self):
        for host in ("0.0.0.0", "::", "192.0.2.1", "localhost.example.invalid"):
            for token in (None, "", "   "):
                with self.subTest(host=host, token=token):
                    env = {"COASTPILOT_LABEL_HOST": host}
                    if token is not None:
                        env["COASTPILOT_LABEL_TOKEN"] = token
                    with patch.dict(os.environ, env, clear=True), patch("flask.Flask.run") as run:
                        with self.assertRaisesRegex(SystemExit, "COASTPILOT_LABEL_TOKEN"):
                            runpy.run_path(str(LABELER_PATH), run_name="__main__")
                        run.assert_not_called()

    def test_non_loopback_bind_with_token_is_explicitly_allowed(self):
        args = self.run_labeler({
            "COASTPILOT_LABEL_HOST": "0.0.0.0",
            "COASTPILOT_LABEL_TOKEN": "unit-test-token",
        })
        self.assertEqual(args["host"], "0.0.0.0")
        self.assertFalse(args["debug"])


class TestLabelerTokenSecurity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.input = root / "input.csv"
        self.output = root / "output.csv"
        self.input.write_text("segment_id,gold_label\nfixture,\n", encoding="utf-8")
        with patch.dict(os.environ, {
            "COASTPILOT_LABEL_INPUT": str(self.input),
            "COASTPILOT_LABEL_OUTPUT": str(self.output),
            "COASTPILOT_LABEL_TOKEN": "unit-test-token",
        }, clear=True):
            module = runpy.run_path(str(LABELER_PATH), run_name="labeler_test")
        self.client = module["app"].test_client()

    def test_missing_or_wrong_token_rejects_all_routes_without_writing(self):
        original = self.input.read_bytes()
        for headers in ({}, {"X-Label-Token": "wrong"}):
            for method, path in (("get", "/"), ("get", "/image?idx=0"), ("post", "/label")):
                with self.subTest(method=method, path=path, headers=headers):
                    response = getattr(self.client, method)(path, headers=headers, data={"idx": "0", "label": "rock"})
                    self.assertEqual(response.status_code, 403)
                    self.assertFalse(self.output.exists())
                    self.assertEqual(self.input.read_bytes(), original)

    def test_valid_header_token_allows_labeling_temporary_csv(self):
        response = self.client.post("/label", headers={"X-Label-Token": "unit-test-token"}, data={"idx": "0", "label": "rock"})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(pd.read_csv(self.output).loc[0, "gold_label"], "rock")

    def test_valid_query_token_allows_browser_labeling(self):
        response = self.client.post("/label?token=unit-test-token", data={"idx": "0", "label": "pebble"})
        self.assertEqual(response.status_code, 302)
        self.assertIn("token=unit-test-token", response.location)
        self.assertEqual(pd.read_csv(self.output).loc[0, "gold_label"], "pebble")


if __name__ == "__main__":
    unittest.main()
