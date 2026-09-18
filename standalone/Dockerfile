# Standalone AppDaemon image for ha-energy-forecast — NOT the HA AppDaemon add-on.
# See ../README.md#standalone-docker-deployment.
#
# Plain Debian/glibc base: AppDaemon and every ML dependency below install as
# prebuilt manylinux wheels straight from PyPI. No compiler, no third-party wheel
# mirror needed — that's only a concern on Alpine/musl base images.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    appdaemon \
    'requests>=2.31.0' \
    'holidays>=0.46' \
    'pandas>=2.0.0' \
    'numpy>=1.24.0' \
    'scikit-learn>=1.4.0' \
    'lightgbm>=4.7.0,<5.0'

# On armv7 (32-bit Raspberry Pi) there's no prebuilt LightGBM wheel on PyPI — drop
# the lightgbm line above if the build fails there. The app falls back to
# scikit-learn's GradientBoostingRegressor automatically; keep libgomp1 either way,
# scikit-learn requires it at import time.

ENTRYPOINT ["appdaemon"]
CMD ["-c", "/conf"]
