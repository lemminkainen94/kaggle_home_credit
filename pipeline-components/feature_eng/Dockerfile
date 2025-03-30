FROM "python:3.11-slim"

WORKDIR /app


RUN pip install --no-cache-dir feature-engine numpy pandas polars scikit-learn

COPY feature_eng /app/

ENTRYPOINT [ "python3", "-u", "feature_eng.py" ]