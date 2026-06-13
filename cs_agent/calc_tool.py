"""Deterministic calculator tool for the CS agent (candidate solution C2).

The model must never do arithmetic in-context. calculate() evaluates a single
arithmetic expression against a safe AST whitelist and returns the exact result,
so fee/reward/NET computations don't drift (e.g. 1499 vs 1500, or summing gross
overcharges instead of the net of overcharges minus undercharges)."""

import ast
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "round": round, "abs": abs, "sum": sum}


def _eval(node):
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression and return its exact numeric result.

    Use this for ALL arithmetic — fees, reward points, percentages, min/max
    caps, and especially net sums (overcharges minus undercharges/missing
    fees). Never compute these in your head; pass the full expression and use
    the returned result verbatim.

    Args:
        expression: A single arithmetic expression using numbers and the
            operators + - * / // % ** with parentheses. The functions min,
            max, round, abs, and sum are allowed. Examples:
            "2.50 + 8.00 + 10.50 + 1.00 + 2.50",
            "min(0.02 * 200.00, 6.00)",
            "round(0.015 * 420000)",
            "sum([2.50, 2.50, 4.00, 1.50]) - 2.50".

    Returns:
        {"result": <number>, "expression": <echo>} on success, or
        {"error": "<reason>"} if the expression cannot be evaluated.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
    except Exception as e:  # noqa: BLE001 - report any parse/eval failure to the model
        return {"error": f"Could not evaluate {expression!r}: {e}"}
    return {"result": result, "expression": expression}
