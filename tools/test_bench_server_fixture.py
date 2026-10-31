#!/usr/bin/env python3.11
# coding=utf-8
"""集成自测：bench_server --fixture + bench_cli（不占 NPU）。

验证 supervisor 的 8 rank 拉起、ctl ACK、结果聚合、进度转发、客户端渲染、干净退出。
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
SERVER = os.path.join(HERE, "bench_server.py")
CLI = os.path.join(HERE, "bench_cli.py")
PY = sys.executable


def _wait_sock(path, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(path)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.05)
    return False


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ds41_bench_fixture_",
                                   dir=os.environ.get("DS41_BENCH_ROOT", "/tmp"))
        cls.job = "fixture_test"
        cls.logdir = os.path.join(cls.tmp, "logs")
        cls.sock = os.path.join(cls.tmp, cls.job, "server.sock")
        cls.proc = subprocess.Popen(
            [PY, "-u", SERVER, "start", "--fixture",
             "--config", "configs/mini_real.yaml",
             "--world", "8", "--cards", "0,1,2,3,4,5,6,7",
             "--logdir", cls.logdir, "--job", cls.job,
             "--mode", "ep8", "--host-cache", "off"],
            cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        # server 的 job 目录由 P.job_root 决定；--job 是 fixture_test，
        # 但 bench_server 的 job_root 默认 /tmp/ds41_bench，除非 DS41_BENCH_ROOT 设置。
        root = os.environ.get("DS41_BENCH_ROOT", "/tmp/ds41_bench")
        cls.sock = os.path.join(root, cls.job, "server.sock")
        if not _wait_sock(cls.sock, 20):
            out = ""
            try:
                cls.proc.kill()
                out = cls.proc.stdout.read()
            except Exception:                       # noqa: BLE001
                pass
            raise RuntimeError(f"fixture server 未就绪：{cls.sock}\n{out[-3000:]}")

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.proc.poll() is None:
                cls.proc.terminate()
                try:
                    cls.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    cls.proc.kill()
        finally:
            shutil.rmtree(cls.tmp, ignore_errors=True)
            try:
                shutil.rmtree(os.path.join(
                    os.environ.get("DS41_BENCH_ROOT", "/tmp/ds41_bench"), cls.job),
                    ignore_errors=True)
            except Exception:                       # noqa: BLE001
                pass

    def _cli(self, *args, timeout=60):
        r = subprocess.run([PY, CLI, *args, "--job", self.job],
                           cwd=BASE, capture_output=True, text=True,
                           timeout=timeout)
        self.assertEqual(r.returncode, 0,
                         f"cli {args} rc={r.returncode}\n"
                         f"stdout:\n{r.stdout[-3000:]}\nstderr:\n{r.stderr[-3000:]}")
        return r.stdout + r.stderr

    def test_1_status(self):
        out = self._cli("status")
        self.assertIn('"ok": true', out)
        # bench_server 默认 --serve-l-max 1024（2026-10-19 常驻口径）
        self.assertIn('"l_pool": 1024', out)

    def test_2_warmup(self):
        out = self._cli("warmup", "--lengths", "128,256,512", "--decode", "1")
        self.assertIn("W1 AC", out)
        self.assertIn("W2 decode", out)
        self.assertIn("W3 shapes", out)
        self.assertIn("预热完成", out)
        self.assertIn("40 层", out)

    def test_3_run(self):
        out = self._cli("run", "--prompt", "你好", "--decode", "4", "--repeat", "2")
        self.assertIn("prefill", out)
        self.assertIn("decode", out)
        self.assertIn("结果（2 次）", out)
        self.assertIn("稳态中位", out)
        self.assertIn("tail -f", out)

    def test_4_stop(self):
        out = self._cli("stop")
        self.assertIn("已请求停止", out)
        t0 = time.time()
        while time.time() - t0 < 15 and self.proc.poll() is None:
            time.sleep(0.1)
        self.assertIsNotNone(self.proc.poll(), "server 未在 stop 后退出")


if __name__ == "__main__":
    unittest.main(verbosity=2)
