# Mechanism verification (ε = 1.0, δ = 1e-05)

Universe: all multisets of size ≤ 4 over [-2, 0, 1, 3]; 140 add/remove neighbour pairs.

| check | passed |
|---|---|
| neighbour generation | PASS |
| sensitivity: count | PASS |
| laplace ε-DP (closed form): count | PASS |
| gaussian (ε,δ)-DP (exact δ): count | PASS |
| sensitivity: clipped sum | PASS |
| laplace ε-DP (closed form): clipped sum | PASS |
| gaussian (ε,δ)-DP (exact δ): clipped sum | PASS |
| sensitivity: centered clipped sum | PASS |
| laplace ε-DP (closed form): centered clipped sum | PASS |
| gaussian (ε,δ)-DP (exact δ): centered clipped sum | PASS |
| sensitivity: histogram | PASS |
| laplace ε-DP (closed form): histogram | PASS |
| gaussian (ε,δ)-DP (exact δ): histogram | PASS |
| laplace ε-DP (numerical grid, supplementary): clipped sum | PASS |
| geometric ε-DP (exact, all integer outputs): count | PASS |

These checks test the implementation on a toy universe. The DP guarantee for arbitrary data comes from the mechanism theorems and sensitivity proofs (KNOWLEDGE.md).
