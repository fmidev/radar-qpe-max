FROM python:3

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y libhdf5-dev

COPY . .

RUN pip install -U pip && pip install --no-cache-dir .

ENV PYART_QUIET=1

ENTRYPOINT ["/usr/local/bin/qpe"]
