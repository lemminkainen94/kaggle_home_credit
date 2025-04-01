FROM "python:3.11-slim"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1
RUN pip install --no-cache-dir hyperopt lightgbm mlflow numpy pandas pyarrow scikit-learn
COPY train /app/

ENTRYPOINT [ "python3", "-u", "train.py" ]