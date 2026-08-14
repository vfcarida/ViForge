# Default ViForge multi-stage Dockerfile
FROM python:3.11-slim

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/

RUN pip install --upgrade pip && pip install -e .

ENTRYPOINT ["viforge"]
CMD ["doctor"]
