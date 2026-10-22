"""Smoke test: load the GRACE student and the GRACE-OFF teacher, evaluate a few
organic molecules with both, and report the force gap the LoRA adapters must close.

Usage:
    python scripts/check_models.py
    python scripts/check_models.py --student /path/to/student_saved_model --teacher /path/to/saved_model
"""

import argparse
import time

import tensorpotential  # noqa: F401  (must come before tensorflow: sets TF_USE_LEGACY_KERAS)
import numpy as np
import tensorflow as tf
from ase.build import molecule
from tensorpotential.calculator import TPCalculator

import config

MOLECULES = ["CH3CH2OH", "C6H6", "CH3COOH", "CH3CONH2", "CH3SH"]


def evaluate(calc, atoms):
    atoms = atoms.copy()
    atoms.calc = calc
    t0 = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default=config.STUDENT_MODEL_DIR, help="student SavedModel")
    parser.add_argument("--teacher", default=config.TEACHER_MODEL_DIR, help="teacher SavedModel")
    args = parser.parse_args()
    config.setup_device()

    print(f"TensorFlow {tf.__version__}, tensorpotential {tensorpotential.__version__}\n")

    student = TPCalculator(model=str(config.require(args.student, "STUDENT_MODEL_DIR")))
    teacher = TPCalculator(model=str(config.require(args.teacher, "TEACHER_MODEL_DIR")))

    print(f"\n{'molecule':<10} {'E_student':>12} {'E_teacher':>12} "
          f"{'|F| teacher':>12} {'F MAE gap':>10}  (eV, eV/A)")
    gaps = []
    for name in MOLECULES:
        atoms = molecule(name)
        atoms.rattle(stdev=0.05, seed=0)  # off-equilibrium so forces are non-trivial
        e_s, f_s, _ = evaluate(student, atoms)
        e_t, f_t, _ = evaluate(teacher, atoms)
        gap = np.abs(f_s - f_t).mean()
        gaps.append(gap)
        print(f"{name:<10} {e_s:12.3f} {e_t:12.3f} "
              f"{np.linalg.norm(f_t, axis=1).mean():12.3f} {gap:10.3f}")

    # Timing after warm-up (first calls include graph compilation)
    _, _, dt = evaluate(teacher, molecule("C6H6"))
    print(f"\nMean force MAE student vs teacher: {np.mean(gaps) * 1000:.0f} meV/A")
    print(f"Teacher eval time (benzene, warm): {dt * 1000:.1f} ms")
    print("\nNote: absolute energies use different per-element references "
          "(PBE-based student vs SPICE wB97M-D3BJ teacher); compare forces. "
          "train_lora.py aligns the energy references before training.")


if __name__ == "__main__":
    main()
