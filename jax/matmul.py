import jax
import jax.numpy as jnp

# JAX arrays live on the accelerator (GPU/TPU) when one is available.
A = jnp.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])  # shape (2, 3)

B = jnp.array([
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0],
])  # shape (3, s2)

# Same API as NumPy: (2, 3) @ (3, 2) -> (2, 2)
C = A @ B

# equivalently: C = jnp.matmul(A, B)
print("A =\n", A)
print("B =\n", B)
print("C = A @ B =\n", C)
print("device:", C.device)