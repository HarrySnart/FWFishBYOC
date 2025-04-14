# Use a Python 3.12 base image
ARG type

FROM python:3.12 AS base

ENV DATASCIENCE_USER datascience
ENV DATASCIENCE_UID 1000
ENV HOME /home/$DATASCIENCE_USER

# Copy the requirements file (if you have one)
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS run-type-local
# nothing to see here

FROM base AS run-type-remote
COPY model_mcmc.py .
CMD ["python", "model_mcmc.py"]

FROM run-type-${type} AS final