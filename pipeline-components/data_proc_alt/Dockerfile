FROM "python:3.11-slim"

WORKDIR /app

RUN pip install --no-cache-dir numpy pandas polars pyarrow scikit-learn
COPY data_proc_alt /app/
COPY kaggle_data /app/data

ENTRYPOINT [ "python3", "-u", "data_proc_alt.py" ]