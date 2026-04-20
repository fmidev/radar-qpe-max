FROM python:3.14-slim AS builder

WORKDIR /build
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends \
        git gcc g++ libproj-dev libgeos-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install build && python -m build --wheel
RUN pip wheel --wheel-dir /build/wheels /build/dist/*.whl

FROM python:3.14-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libproj25 libgeos-c1v5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels/ /tmp/wheels/
RUN pip install --no-index --no-deps /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels/

ENV PYART_QUIET=1

ENTRYPOINT ["/usr/local/bin/qpe"]
