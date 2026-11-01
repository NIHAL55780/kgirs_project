"""Safe arithmetic evaluator (no eval()): numbers, + - * / // % **, parentheses and a few math functions."""
import ast
import math
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
_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": lambda *a: sum(a),
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}

DESCRIPTION = (
    "calculator[expression]: evaluates an arithmetic expression, e.g. calculator[2024 - 1867] "
    "or calculator[round(1500 / 7, 2)]. Supports + - * / // % ** ( ) and abs, round, min, max, "
    "sum, sqrt, log, log10, exp, floor, ceil."
)


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    expr = expression.strip().replace("^", "**").replace(",", "").replace("×", "*").replace("÷", "/")
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except Exception as exc:  # noqa: BLE001
        return f"Calculator error: {exc}. Use a plain arithmetic expression like 2024 - 1867."
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression.strip()} = {result}"
