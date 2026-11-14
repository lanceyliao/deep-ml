from math import prod


def term_bounds(terms: dict[str, int], ranges: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """计算线性表达式在半开区间 [lo, hi) 下的精确最小与最大值 (vmin, vmax)。"""
    vmin = vmax = terms.get("", 0)
    for var, coeff in terms.items():
        if var == "":
            continue
        lo, hi = ranges[var]
        if coeff > 0:
            vmin += coeff * lo
            vmax += coeff * (hi - 1)
        elif coeff < 0:
            vmin += coeff * (hi - 1)
            vmax += coeff * lo
    return vmin, vmax


def fold_div(
    terms: dict[str, int], c: int, ranges: dict[str, tuple[int, int]]
) -> tuple[dict[str, int], dict[str, int] | None]:
    """
    计算 floor(expr / c) (c > 0)。
    - 系数能被 c 整除的项（含常数项）除以 c 后进入 quotient；
    - 不能整除的项保持原系数进入 remainder；
    - 两个字典均将 "" 放在首位；若 remainder 严格落在 [0, c) 内，则返回 (quotient, None)。
    """
    quotient: dict[str, int] = {"": 0}
    remainder: dict[str, int] = {"": 0}

    for var, coeff in terms.items():
        if coeff == 0:
            continue
        if coeff % c == 0:
            quotient[var] = coeff // c
        else:
            remainder[var] = coeff

    r_min, r_max = term_bounds(remainder, ranges)
    if 0 <= r_min and r_max < c:
        return quotient, None
    return quotient, remainder


def fold_mod(
    terms: dict[str, int], c: int, ranges: dict[str, tuple[int, int]]
) -> tuple[dict[str, int], bool]:
    """
    计算 expr % c (c > 0)。
    - 丢弃系数能被 c 整除的项（含常数项）；
    - 其余项保持原输入顺序保留；若 "" 被丢弃，则在末尾补上 "": 0；
    - exact 表示余数边界是否严格落在 [0, c) 内。
    """
    remainder: dict[str, int] = {}

    for var, coeff in terms.items():
        if coeff != 0 and coeff % c != 0:
            remainder[var] = coeff

    if "" not in remainder:
        remainder[""] = 0

    r_min, r_max = term_bounds(remainder, ranges)
    exact = 0 <= r_min and r_max < c
    return remainder, exact


def unflatten(
    shape: tuple[int, ...], flat_terms: dict[str, int], ranges: dict[str, tuple[int, int]]
) -> list[dict[str, int] | str]:
    """
    按行主序（row-major）步长，对原始 flat_terms 逐轴计算 (flat // s_j) % shape_j。
    只有当除法无残余项（rem is None）且取模精确（exact is True）时，才记录线性表达式。
    """
    results: list[dict[str, int] | str] = []
    # 计算每个轴的 row-major 步长：s_j = prod(shape[j+1:])
    strides = [prod(shape[j + 1 :]) for j in range(len(shape))]

    for s_j, dim_j in zip(strides, shape):
        quot, rem = fold_div(flat_terms, s_j, ranges)
        mod_expr, exact = fold_mod(quot, dim_j, ranges)

        if rem is None and exact:
            results.append(mod_expr)
        else:
            results.append("div/mod")

    return results