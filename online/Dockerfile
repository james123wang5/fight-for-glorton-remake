FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    GLORTON_ASSET_SCALE=1

WORKDIR /app

COPY requirements.txt requirements-online.txt ./
RUN pip install --no-cache-dir -r requirements-online.txt

COPY . .
RUN python tools/install_runtime_assets.py

EXPOSE 10000
CMD ["sh", "-c", "python -m online.server --host 0.0.0.0 --port ${PORT:-10000}"]
