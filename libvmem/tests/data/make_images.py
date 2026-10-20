#!/usr/bin/env python3
# makes the small grayscale base images (PGM) that the stress test (stress.cu) tiles up to VRAM-sized images
# python3 tests/data/make_images.py   (needs numpy; the photo needs matplotlib and pillow)
import os
import numpy as np

out = os.path.dirname(os.path.abspath(__file__))
n = 1024
y, x = np.mgrid[0:n, 0:n].astype(np.float64)


def save(name, a):
    a = np.clip(a, 0, 255).astype(np.uint8)
    with open(os.path.join(out, name + ".pgm"), "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (a.shape[1], a.shape[0]))
        f.write(a.tobytes())


r = np.hypot(x - n / 2, y - n / 2)
save("rings", 127 + 100 * np.sin(r / 7) + 28 * (x / n))

rng = np.random.default_rng(1)
save("checker", 200 * (((x // 64) + (y // 64)) % 2) + 30 + rng.normal(0, 12, (n, n)))

save("plasma", 128 + 32 * (np.sin(x / 31) + np.sin(y / 17) + np.sin((x + y) / 23) + np.sin(r / 19)))

c = (x - 0.7 * n) / (n * 0.35) + 1j * (y - n / 2) / (n * 0.35)
z = np.zeros_like(c)
it = np.zeros((n, n))
for i in range(120):
    m = np.abs(z) <= 2
    z[m] = z[m] ** 2 + c[m]
    it[m] += 1
save("mandelbrot", it * 255 / 120)

try:  # a real photo: matplotlib ships grace_hopper.jpg (public domain, US Navy)
    import matplotlib
    from PIL import Image
    p = os.path.join(matplotlib.get_data_path(), "sample_data", "grace_hopper.jpg")
    save("photo", np.asarray(Image.open(p).convert("L")))
except Exception as e:
    print("no photo:", e)
