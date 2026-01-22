FROM --platform=linux/arm64/v8 public.ecr.aws/awsguru/aws-lambda-adapter:0.8.3 AS lwa

FROM --platform=linux/arm64/v8 public.ecr.aws/lambda/python:3.13

# Install AWS Lambda Web Adapter extension
COPY --from=lwa /lambda-adapter /opt/extensions/lambda-adapter

# Install uv package manager following official guidance
RUN pip install --no-cache-dir uv

WORKDIR /var/task

# Copy project metadata and install dependencies using the lockfile for reproducibility
COPY pyproject.toml ./
COPY README.md ./README.md
COPY requirements.lock ./requirements.lock
RUN cp requirements.lock uv.lock

# Copy application source (editable install target)
COPY src/ ./src

RUN uv sync --frozen

# Expose application via uvicorn (Lambda Web Adapter handles the HTTP bridge)
ENTRYPOINT ["/var/task/.venv/bin/python", "-m", "uvicorn"]
CMD ["data_bolt.app:app", "--host", "0.0.0.0", "--port", "8080"]
