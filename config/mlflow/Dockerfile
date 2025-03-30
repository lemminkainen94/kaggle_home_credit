# Defining base image
FROM python:3.11-slim
# Installing MLflow from PyPi
RUN pip install mlflow[extras]==2.17.2
# Defining start up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]