"""Check the simulator (paper Section 2.6). Run from the repo root: python code/validate.py > results/validation.txt
(a) Reproduce Rath et al. (arXiv:2603.27278) Table I: no noise, 50,000 sifted bits per trial, 50 trials.
(b) Check QBER = e + (f/4)(1-2e) with channel noise.
(c) Cross-check the physics against Qiskit circuits.
"""
import numpy as np
from bb84 import run_bb84, qber_theory

rng = np.random.default_rng(42)
print("(a) Rath Table I check: no noise, ~50,000 sifted bits, 50 trials")
print(f"{'f':>4} {'mean':>7} {'std':>7} {'95% CI':>18} {'theory':>7} {'Rath':>7}")
rath = {0.0: 0.000, 0.1: 0.025, 0.2: 0.050, 0.5: 0.124, 1.0: 0.250}
rows_a = []
for f, r in rath.items():
    q = np.array([run_bb84(100_000, f, 0.0, rng)["qber"] for _ in range(50)])
    m, s = q.mean(), q.std(ddof=1)
    lo, hi = m - 1.96 * s / np.sqrt(50), m + 1.96 * s / np.sqrt(50)
    rows_a.append((f, m, s, lo, hi))
    print(f"{f:4.1f} {m:7.4f} {s:7.4f}   [{lo:.4f}, {hi:.4f}] {f/4:7.3f} {r:7.3f}")

print("\n(b) With channel noise, 4096 qubits, all sifted bits, 2000 trials")
print(f"{'e':>5} {'f':>4} {'sim':>7} {'theory':>7}")
for e in [0.02, 0.05, 0.10]:
    for f in [0.0, 0.3, 1.0]:
        q = np.mean([run_bb84(4096, f, e, rng)["qber"] for _ in range(2000)])
        print(f"{e:5.2f} {f:4.1f} {q:7.4f} {qber_theory(f, e):7.4f}")

print("\n(c) Qiskit cross-check (no noise): one circuit per qubit, Eve measures mid-circuit")
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
sim = AerSimulator()
qrng = np.random.default_rng(7)
for f in [0.0, 0.5, 1.0]:
    N = 16000
    circs, info = [], []
    for _ in range(N):
        ab, aB, bB = qrng.integers(0, 2, 3)
        eve = qrng.random() < f
        eB = qrng.integers(0, 2)
        qc = QuantumCircuit(1, 2)
        if ab: qc.x(0)
        if aB: qc.h(0)
        if eve:
            if eB: qc.h(0)
            qc.measure(0, 0)          # Eve measures in a random basis
            if eB: qc.h(0)            # and resends the result
        if bB: qc.h(0)
        qc.measure(0, 1)              # Bob
        circs.append(qc); info.append((ab, aB, bB))
    res = sim.run(circs, shots=1, seed_simulator=11).result()
    err = n = 0
    for i, (ab, aB, bB) in enumerate(info):
        if aB != bB: continue
        bit = int(list(res.get_counts(i).keys())[0][0])  # clbit 1 is leftmost
        n += 1; err += bit != ab
    print(f"f={f:.1f}: Qiskit QBER {err/n:.4f} on {n} sifted bits (theory {f/4:.3f})")
