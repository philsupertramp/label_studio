FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch to bypass massive CUDA binaries
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir boto3 datasets Pillow transformers requests

COPY auto_label.py .

ENTRYPOINT ["python", "auto_label.py"]
