# Default ViForge multi-stage Dockerfile
FROM python:3.11-slim

# Create non-root system user and group
RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --create-home --shell /bin/bash appuser

WORKDIR /workspace

COPY pyproject.toml README.md /workspace/
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/

RUN pip install --upgrade pip && \
    pip install -e . && \
    chown -R appuser:appgroup /workspace

USER appuser

ENTRYPOINT ["viforge"]
CMD ["doctor"]

