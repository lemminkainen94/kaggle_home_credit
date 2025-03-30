FROM "python:3.11-slim"

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas polars pyarrow
COPY data_proc /app/
COPY kaggle_data /app/data

ENTRYPOINT [ "python3", "-u", "data_proc.py" ]