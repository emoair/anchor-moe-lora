ARG NODE_IMAGE=docker.io/library/node@sha256:a149cd71dccd68704a07d4e4ca3e610c27301852b0f556865cfdb6e2856f8bed
ARG PYTHON_IMAGE=docker.io/library/python@sha256:5c5e0496473632460861e691a03cce82205c38556d9c0be4e6cb5915380f1e50
FROM ${NODE_IMAGE} AS node_runtime
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="Anchor train sandbox"
LABEL org.opencontainers.image.description="Digest-pinned Python and Node validation image for train-only execution evidence"
LABEL org.opencontainers.image.source="https://github.com/emoair/anchor-moe-lora"

# The full official Python Bookworm image already carries Git and the common
# build toolchain. Copy only the Node runtime and package-manager payload from
# the independently digest-pinned official Node image; do not overwrite
# Python's /usr/local tree and do not contact a package mirror during build.
COPY --from=node_runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=node_runtime /usr/local/lib/node_modules /usr/local/lib/node_modules

COPY scripts/tooling/train_sandbox_validate.py /usr/local/bin/anchor-validate
RUN chmod 0755 /usr/local/bin/anchor-validate \
    && ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && ln -sf ../lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack \
    && node --version \
    && npm --version \
    && python --version \
    && git --version \
    && anchor-validate --version
