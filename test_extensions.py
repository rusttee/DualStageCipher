from __future__ import annotations
import numpy as np
from common import CipherConfig
from cipher_core import build_plan, encrypt_array, decrypt_array, encrypt_with_plan, decrypt_with_plan


def main():
    rng = np.random.default_rng(20261006)
    cfg = CipherConfig(discard_first_100=True)
    for shape in [(32, 48), (32, 48, 3), (48, 32, 3)]:
        a = rng.integers(0, 256, size=shape, dtype=np.uint8)
        c, _ = encrypt_array(a, cfg)
        d = decrypt_array(c, cfg, a.shape[:2])
        assert np.array_equal(a, d), shape
        plan = build_plan(cfg, *a.shape[:2])
        c2 = encrypt_with_plan(a, plan)
        assert np.array_equal(c, c2)
        assert np.array_equal(a, decrypt_with_plan(c2, plan))
    print("paper-locked extension core tests passed")


if __name__ == "__main__":
    main()
