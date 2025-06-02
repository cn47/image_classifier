ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY --chown=1000:1000 ./pyproject.toml pyproject.toml

RUN uv pip install -r pyproject.toml --all-extras
