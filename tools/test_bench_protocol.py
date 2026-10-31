#!/usr/bin/env python3.11
# coding=utf-8
"""最小单测：bench_protocol 的行解析 / 结果解析 / socket JSONL。"""
import io
import json
import os
import socket
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_protocol as P


class TestParse(unittest.TestCase):
    def test_stage_lines(self):
        cases = {
            "  [W1  1/40] layer 0 win T=128 算子 24": ("W1", 1, 40),
            "  [W2 40/40] layer 39 win T=1 算子 24": ("W2", 40, 40),
            "  [W3 2/5] L=512 1.02s": ("W3", 2, 5),
            "  [P→D 33/40] TP all2all(fused)：": ("P→D", 33, 40),
            "  [D→P 28/40] 回载 EP raw：": ("D→P", 28, 40),
        }
        for line, want in cases.items():
            got = P.parse_line(line)
            self.assertIsNotNone(got, line)
            self.assertEqual((got["stage"], got["cur"], got["total"]), want, line)

    def test_layer_and_inc(self):
        got = P.parse_line("  [ 12/40] layer 11 win T=512 算子 24")
        self.assertEqual(got["stage"], "prefill")
        self.assertEqual((got["cur"], got["total"]), (12, 40))
        got = P.parse_line("  [inc 3/32] 预测位置 515 → id=396 本步 0.69s")
        self.assertEqual(got["stage"], "decode")
        self.assertEqual((got["cur"], got["total"]), (3, 32))

    def test_hostcache(self):
        got = P.parse_line("[moe-host-cache] 2000/92160  12.3/268.9 GiB  1500 MiB/s")
        self.assertEqual(got["stage"], "hostcache")
        self.assertEqual((got["cur"], got["total"]), (2000, 92160))
        self.assertIn("1500 MiB/s", got["extra"])

    def test_result_and_evt(self):
        r = P.parse_result('@@RESULT {"ok": true, "prefill_s": 1.02, "ids": [1,2]}')
        self.assertEqual(r["prefill_s"], 1.02)
        self.assertEqual(r["ids"], [1, 2])
        e = P.parse_evt('@@EVT {"type":"progress","cur":1}')
        self.assertEqual(e["type"], "progress")
        self.assertIsNone(P.parse_result("not a result"))
        self.assertIsNone(P.parse_line("plain log line"))


class TestStageBar(unittest.TestCase):
    def test_transitions(self):
        out = io.StringIO()
        bar = P.StageBar(stream=out, enabled=True, width=8, every=0)
        for line in [
            "  [W1  1/40] layer 0",
            "  [W1 40/40] layer 39",
            "  [W2  1/40] layer 0",
            "  [W2 40/40] layer 39",
            "  [W3 1/2] L=128 1.0s",
            "  [W3 2/2] L=256 1.1s",
            "  [ 1/40] layer 0",
            "  [ 40/40] layer 39",
            "  [inc 1/4] ...",
            "  [inc 4/4] ...",
            "  [P→D 1/40] ...",
            "  [P→D 40/40] ...",
            "  [D→P 1/40] ...",
            "  [D→P 40/40] ...",
        ]:
            self.assertTrue(bar.feed(line), line)
        bar.close()
        text = out.getvalue()
        for stage in ("W1 AC", "W2 decode", "W3 shapes", "prefill", "decode",
                      "P→D", "D→P"):
            self.assertIn(stage, text)

    def test_ignore_unknown(self):
        bar = P.StageBar(stream=io.StringIO(), enabled=False, every=0)
        self.assertFalse(bar.feed("hello"))


class TestSocketJsonl(unittest.TestCase):
    def test_roundtrip(self):
        a, b = socket.socketpair()
        P.send_json(a, {"op": "run", "text": "你好", "n": 3})
        jl = P.JsonLines(b)
        msg = jl.recv()
        self.assertEqual(msg["text"], "你好")
        self.assertEqual(msg["n"], 3)
        jl.close()
        a.close()
        b.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
