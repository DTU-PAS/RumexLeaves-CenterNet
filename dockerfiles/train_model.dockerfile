# Base image
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3.8 python3-pip python3-setuptools gcc libgl1 libglib2.0-0 && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml /code/pyproject.toml
COPY rumexleaves_centernet/ /code/rumexleaves_centernet/
COPY submodules/ /code/submodules/

WORKDIR /
#--no-cache-dir
RUN pip3 install -r requirements.txt
RUN pip3 install code/. --no-deps
RUN pip3 install code/submodules/Annotation-Converter/. --no-deps

ENTRYPOINT ["python3", "-u", "code/rumexleaves_centernet/tools/train.py", "--exp_file", "exp_file.py", "--logger", "tensorboard"]
