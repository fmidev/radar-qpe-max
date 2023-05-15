FROM python:3

WORKDIR /usr/src/app

COPY . .

RUN pip install -U pip && pip install --no-cache-dir .

ENV PYART_QUIET=1

ENTRYPOINT ["/usr/local/bin/sademaksit"]
