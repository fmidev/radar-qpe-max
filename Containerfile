# Containerfile for qpemax - QPE statistical indicators over moving temporal windows

FROM python:3.14-slim AS builder

# Build-time system deps (git for hatch-vcs + git+https deps; compilers/headers
# for any C extensions that lack matching manylinux wheels).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ libproj-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv we can copy verbatim into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Copy source (include .git so hatch-vcs can derive the version).
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY .git/ .git/

# Install the project and all its runtime dependencies into /opt/venv.
RUN pip install .


FROM python:3.14-slim

LABEL org.opencontainers.image.title="sademaksit"
LABEL org.opencontainers.image.description="QPE statistical indicators over moving temporal windows"
LABEL org.opencontainers.image.source="https://github.com/fmidev/radar-qpe-max"

# Runtime shared libraries only (no compilers, no git).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 libproj25 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYART_QUIET=1

ENTRYPOINT ["qpe"]
