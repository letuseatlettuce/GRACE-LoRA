"""Low-rank adapters (LoRA) for GRACE weights:  W = W0 + (alpha / rank) * B @ A.

The adapters live outside the GRACE model. Each training step:
  1. tensorpotential computes dL/dW for the (merged) weight W as usual,
  2. the chain rule maps it onto the factors:  dL/dB = s * G A^T,  dL/dA = s * B^T G,
  3. the optimizer updates A and B,
  4. W is overwritten with W0 + s * B @ A.
This is mathematically identical to LoRA, needs no changes to GRACE itself, and keeps the
model rotationally equivariant: only weight values change, never the architecture.

Per-element adaptation: GRACE's `reducing_` weights have shape
(n_elements, n_out, n_in_channels, n_paths). Each selected element z gets its own adapter on
the matrix W[z] of shape (n_out, n_in_channels * n_paths); other elements stay frozen.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class LoRAAdapter:
    def __init__(self, var: tf.Variable, rank: int, alpha: float,
                 element_rows: list[int] | None = None, seed: int = 0):
        self.var = var
        self.name = var.name.split(":")[0]
        self.shape = var.shape.as_list()
        self.w0 = tf.constant(var.numpy())  # frozen base weights
        self.rank = rank
        self.scale = alpha / rank

        if element_rows is not None:
            # one adapter per selected element: W[z] -> (n_out, rest)
            self.rows = tf.constant(np.asarray(element_rows, dtype=np.int32)[:, None])
            n_batch, n_out = len(element_rows), self.shape[1]
            n_in = int(np.prod(self.shape[2:]))
            self.block_shape = [n_batch] + self.shape[1:]
        else:
            # whole tensor as one matrix: (shape[0], rest)
            self.rows = None
            n_batch, n_out = 1, self.shape[0]
            n_in = int(np.prod(self.shape[1:]))
            self.block_shape = self.shape

        dtype = var.dtype
        rng = tf.random.Generator.from_seed(seed)
        # Standard LoRA init: A random, B zero  ->  W == W0 at step 0
        self.A = tf.Variable(rng.normal([n_batch, rank, n_in], stddev=1.0 / np.sqrt(n_in),
                                        dtype=dtype), name=f"{self.name}/lora_A")
        self.B = tf.Variable(tf.zeros([n_batch, n_out, rank], dtype=dtype),
                             name=f"{self.name}/lora_B")

    @property
    def factors(self):
        return [self.A, self.B]

    @property
    def n_params(self):
        return int(tf.size(self.A)) + int(tf.size(self.B))

    def delta(self):
        d = self.scale * tf.matmul(self.B, self.A)  # (n_batch, n_out, n_in)
        d = tf.reshape(d, self.block_shape)
        if self.rows is not None:
            d = tf.scatter_nd(self.rows, d, self.shape)
        return d

    def factor_grads(self, grad_w):
        """Chain rule: gradient w.r.t. the merged weight -> gradients w.r.t. A and B."""
        with tf.GradientTape() as tape:
            d = self.delta()
        return tape.gradient(d, self.factors, output_gradients=grad_w)

    def merge(self):
        """Write W0 + delta into the model weight."""
        self.var.assign(self.w0 + self.delta())

    def state(self):
        return {f"{self.name}/A": self.A.numpy(), f"{self.name}/B": self.B.numpy()}

    def load_state(self, state):
        self.A.assign(state[f"{self.name}/A"])
        self.B.assign(state[f"{self.name}/B"])


def build_adapters(variables, rank, alpha, n_elements, element_rows, seed=0):
    """Create adapters; tensors whose first axis is the element axis get per-element LoRA."""
    adapters = []
    for i, var in enumerate(variables):
        per_element = var.shape.rank >= 3 and var.shape[0] == n_elements
        adapters.append(LoRAAdapter(var, rank, alpha,
                                    element_rows=element_rows if per_element else None,
                                    seed=seed + i))
    return adapters
