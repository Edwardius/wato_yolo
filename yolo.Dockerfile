FROM pure/python:3.8-cuda10.2-base
WORKDIR /project

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    gcc \
    ffmpeg \
    libsm6 \
    libxext6 \
	gfortran \
	libopenblas-dev \
	liblapack-dev

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m pip install pip --upgrade
RUN pip install \  
	matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    opencv-python>=4.1.1 \
    Pillow>=7.1.2 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    torch>=1.7.0 \
    torchvision>=0.8.1 \
    tqdm>=4.41.0 \
    # Logging -------------------------------------
    tensorboard>=2.4.1 \
    wandb \
    # Plotting ------------------------------------
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    # Export --------------------------------------
    # coremltools>=4.1  # CoreML export
    # onnx>=1.9.0  # ONNX export
    # onnx-simplifier>=0.3.6  # ONNX simplifier
    # scikit-learn==0.19.2  # CoreML quantization
    # tensorflow>=2.4.1  # TFLite export
    # tensorflowjs>=3.9.0  # TF.js export
    # openvino-dev  # OpenVINO export
    # Extras --------------------------------------
    # albumentations>=1.0.3
    # Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
    # pycocotools>=2.0  # COCO mAP
    # roboflow
    thop 

# RUN pip install --pre 'torch==1.10.0.dev20210921+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install pytorchvideo
RUN pip install torch pytorchvideo torchvision
# RUN pip install --pre 'torchvision==0.11.0.dev2021092+cu111' -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html
# RUN pip install torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 pytorchvideo -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install easydict tensorboardx

WORKDIR /project/yolov5

# python3 ./main.py --config configs/ROAD/SLOWFAST_R50_ACAR_HR2O.yaml
