FROM python:3.11-slim-bookworm
RUN apt update
RUN apt install -y libexpat1-dev
COPY requirements.txt /tmp/pip-tmp/
RUN pip install -r /tmp/pip-tmp/requirements.txt