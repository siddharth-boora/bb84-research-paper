"""BB84 simulated qubit by qubit, with partial intercept-resend and channel bit-flip noise."""
import numpy as np


def run_bb84(n_qubits, f, e, rng, check_fraction=None):
    """Run one BB84 session.

    n_qubits       qubits Alice sends
    f              fraction Eve intercepts (0..1)
    e              channel bit-flip probability (0..0.5)
    check_fraction share of sifted bits revealed to estimate QBER
                   (None = use all sifted bits)
    Returns dict with sifted length, check size, errors, QBER estimate.
    """
    a_bit = rng.integers(0, 2, n_qubits)
    a_basis = rng.integers(0, 2, n_qubits)

    # The state travelling to Bob: (bit, basis). Starts as Alice's.
    s_bit = a_bit.copy()
    s_basis = a_basis.copy()

    # Eve intercepts a random subset.
    eve = rng.random(n_qubits) < f
    e_basis = rng.integers(0, 2, n_qubits)
    wrong = eve & (e_basis != a_basis)
    # Same basis: Eve reads Alice's bit. Wrong basis: 50/50 outcome.
    e_bit = np.where(wrong, rng.integers(0, 2, n_qubits), a_bit)
    s_bit = np.where(eve, e_bit, s_bit)
    s_basis = np.where(eve, e_basis, s_basis)

    # Bob measures.
    b_basis = rng.integers(0, 2, n_qubits)
    b_bit = np.where(b_basis == s_basis, s_bit, rng.integers(0, 2, n_qubits))

    # Channel noise flips Bob's result with probability e.
    flip = rng.random(n_qubits) < e
    b_bit = b_bit ^ flip

    # Sifting.
    keep = a_basis == b_basis
    a_sift, b_sift = a_bit[keep], b_bit[keep]
    k = a_sift.size

    if check_fraction is None:
        idx = np.arange(k)
    else:
        m = int(check_fraction * k)
        idx = rng.choice(k, size=m, replace=False)
    m = idx.size
    errors = int(np.sum(a_sift[idx] != b_sift[idx]))
    return {"sifted": k, "check": m, "errors": errors,
            "qber": errors / m if m else float("nan")}


def qber_theory(f, e):
    """Expected QBER: noise plus Eve, with flips that can cancel."""
    return e + (f / 4) * (1 - 2 * e)


def f_max(e, limit=0.11):
    """Largest intercept fraction that keeps expected QBER at or below `limit`."""
    return min(1.0, max(0.0, 4 * (limit - e) / (1 - 2 * e)))
