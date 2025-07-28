FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
COPY rp_handler.py /
COPY mnist_image_classifier.py /

# Start the container
CMD ["python3", "-u", "rp_handler.py"]