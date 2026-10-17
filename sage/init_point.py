"""Reference-based initialisation (paper Sec. 3.5, Eq. 9)."""


def blend(x_o, x_ref, beta):
    return ((1 - beta) * x_o + beta * x_ref).clamp(0, 1)


def reference_init(x_o, refs, oracle, cfg, info):
    """Verify references in the fixed shared order; bisect on [x_o, first real reference].

    x_o is known to be fake-classified from clean screening (not charged). Returns the real
    endpoint of the bracket, or None (initialisation failure).
    """
    c = cfg["init"]
    ref = None
    for i, x_ref in enumerate(refs[: c["max_references"]]):
        if (yield from oracle.query(x_ref)):
            ref, info["init_ref_index"] = x_ref, i
            break
    if ref is None:
        info["init_failure"] = True
        return None
    lo, hi, n = 0.0, 1.0, 0
    while hi - lo > c["bisection_tol"] and n < c["bisection_max_queries"]:
        mid = 0.5 * (lo + hi)
        n += 1
        if (yield from oracle.query(blend(x_o, ref, mid))):
            hi = mid
        else:
            lo = mid
    info["init_failure"] = False
    info["init_beta"] = hi
    info["init_queries"] = oracle.count
    return blend(x_o, ref, hi)
