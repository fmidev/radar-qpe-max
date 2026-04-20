# Containerfile for qpemax - QPE statistical indicators over moving temporal windows

FROM python:3.14-slim AS builder

# Install build dependencies (C extensions needed for dep pre-compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ libproj-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --no-cache-dir hatch hatch-vcs

# Copy source for building (include .git for hatch-vcs versioning)
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY .git/ .git/

# Build project wheel and pre-compile all dependencies
# hatchling included so the final stage can build radproc from its git+ URL
RUN hatch build -t wheel
RUN pip wheel --wheel-dir /build/wheels hatchling hatch-vcs /build/dist/*.whl


FROM python:3.14-slim

LABEL org.opencontainers.image.title="sademaksit"
LABEL org.opencontainers.image.description="QPE statistical indicators over moving temporal windows"
LABEL org.opencontainers.image.source="https://github.com/fmidev/radar-qpe-max"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libproj25 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels/ qpemax \
    && rm -rf /tmp/wheels/

ENV PYART_QUIET=1

ENTRYPOINT ["/usr/local/bin/qpe"]
