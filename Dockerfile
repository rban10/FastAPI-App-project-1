# YOLO11n instance segmentation: FastAPI service + Streamlit UI.
# The same image runs either process; the default command starts the API.
#
#   API: docker run -p 8000:8000 <image>
#   UI:  docker run -p 8501:8501 -e API_URL=http://<api-host>:8000 <image> \
#          streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics

# ultralytics pulls in the full opencv-python, which needs these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of PyTorch, installed first so ultralytics does not pull the much larger CUDA build.
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.ultralytics \
    && chown -R appuser:appuser /app
USER appuser

# Bake the weights into the image so containers start without a network download.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n-seg.pt')"

COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser streamlit_app.py .

ENV MODEL_PATH=/app/yolo11n-seg.pt

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
