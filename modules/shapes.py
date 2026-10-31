"""符号形状求值器（白名单 AST，无 eval）。

支持：
  · 形状 `[T,H]`、`[T//ratio+1,512]`、`[1+nblk,128,1,64]`；
  · 标量 `1+nblk`、`min(8192, total_q)`、`2I`（隐式乘法）、`H//2`；
  · 函数 `min/max`；运算 `+ - * // % ( )`。

未知符号抛 `ShapeError`（导出器把它列入诊断，不让整棵树崩）。
JS 侧有一份同构实现（`tools/model_viewer.template.html`），用导出器生成的
`shape_vectors` 对拍。
"""
from __future__ import annotations

import ast
import functools
import re

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
            ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub,
            ast.Call, ast.Load, ast.Tuple, ast.List)
_FUNCS = {"min": min, "max": max}
_IMPLICIT = re.compile(r"(\d)([A-Za-z_])")


class ShapeError(Exception):
    pass


@functools.lru_cache(maxsize=4096)
def _parse(expr: str):
    src = _IMPLICIT.sub(r"\1*\2", expr.strip())
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ShapeError(f"无法解析形状 {expr!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ShapeError(f"形状 {expr!r} 含不允许的语法 {type(node).__name__}")
    return tree


def eval_expr(expr: str, env: dict) -> int:
    tree = _parse(expr)
    return int(_eval(tree.body, env))


def _eval(node, env):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int):
            raise ShapeError(f"形状常量必须是整数：{node.value!r}")
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ShapeError(f"未知符号 {node.id!r}")
        v = env[node.id]
        if isinstance(v, (list, tuple)):
            raise ShapeError(f"符号 {node.id!r} 是列表，不能参与算术")
        return int(v)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.FloorDiv):
            if b == 0:
                raise ShapeError("除零")
            return a // b
        if isinstance(node.op, ast.Mod):
            if b == 0:
                raise ShapeError("除零")
            return a % b
        raise ShapeError(f"不支持的运算 {type(node.op).__name__}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, env)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ShapeError(f"不支持的函数 {ast.unparse(node.func)}")
        args = [_eval(a, env) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ShapeError(f"不支持的表达式 {ast.unparse(node)}")


def eval_shape(shape: str | None, env: dict):
    """`"[T,H]"` → `[8192, 5120]`；`"1+nblk"` → `65`；None → None。"""
    if not shape:
        return None
    s = shape.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [eval_expr(part, env) for part in _split_top(inner)]
    return eval_expr(s, env)


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def shape_symbols(shape: str | None) -> set[str]:
    """抽出形状表达式里的符号名（诊断用）。"""
    if not shape:
        return set()
    try:
        tree = _parse(shape.strip().strip("[]"))
    except ShapeError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
