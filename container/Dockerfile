# The measurement, as an image.
#
#   docker build -f container/Dockerfile \
#       --build-arg SUITE_REVISION=$(git rev-parse HEAD) \
#       --build-arg IMAGE_REFERENCE=<registry>/<name>:<tag> \
#       -t <registry>/<name>:<tag> .
#
# WHY.  `bin/bootstrap_measure.sh` is the specification of this file: every
# rental used to run it cold and pay `stage setup ok 4m 02s` of apt, pip and
# git before a single byte of the artifact was fetched.  It is baked here
# instead -- and the SAME script still runs at container start, where every one
# of its steps is guarded and therefore a no-op, so the recipe has exactly one
# definition and the image cannot drift from it.
#
# BASE IMAGE.  Deliberately plain `ubuntu:24.04`, not a vendor's pytorch
# template.  Two runs of another agent died on the RunPod pytorch image, for
# two reasons that are properties of that image rather than of this work: it
# ships torch WITHOUT numpy, and its system python is PEP 668
# externally-managed so `pip install` refuses.  A clean base plus the pinned
# wheel set has neither problem, and 24.04 ships python3.12 -- which the
# bootstrap asserts, because the proven environment recipe is py3.12-only.
# torch's cu130 wheels carry their own CUDA userspace; only libcuda.so comes
# from the host, through the container runtime.
#
# PINS.  This project's product is reproducible numbers, so nothing floats:
# the wheel set, the pipeline commit and its patch series all come from
# bootstrap_measure.sh, and the build writes what it actually resolved into
# /opt/fidelity/BUILD.json.  An image that silently moved its torch version
# would silently change the arithmetic -- and we have measured that even the
# GPU MODEL shifts the result (docs/ARCHITECTURE-DETERMINISM.md).
# Independent overlay target: BuildKit can reuse this published numerical stack
# without evaluating the base/ssh build stages.
ARG HF_JOBS_BASE_IMAGE=ghcr.io/malaiwah/quant-fidelity-measure@sha256:b587227adb3dc56ca2f6fcbd6ecabb143fed04d3d7fabd2d76cbf9bf92bb8006
FROM ubuntu:24.04 AS base

ARG SUITE_REVISION=""
ARG IMAGE_REFERENCE=""
ARG DEBIAN_FRONTEND=noninteractive

# PLATFORM CONTRACT. This exact Python 3.12/CUDA 13.0 measurement image supports
# linux/amd64 only. NVIDIA's pinned cuSPARSELt 0.8.0 ARM wheel has invalid
# internal platform metadata; pip check correctly rejects it. Do not bypass
# that check, repack upstream binaries, or change CUDA versions implicitly.
# See docs/CONTAINER.md for the upstream evidence and ARM re-enablement boundary.
#
# Fail before apt or pip, including for raw Docker/Podman builds outside CI.
# TARGETARCH is automatic under BuildKit; dpkg covers builders without it and
# prevents a caller-supplied build argument from misidentifying the base image.
ARG TARGETARCH
RUN arch="$(dpkg --print-architecture)" \
 && case "${TARGETARCH:-$arch}/$arch" in \
      amd64/amd64) ;; \
      *) echo "Unsupported CUDA measurement target ${TARGETARCH:-unset}/$arch: use linux/amd64. The exact ARM closure fails upstream cuSPARSELt wheel validation; see docs/CONTAINER.md. No CPU or alternate-CUDA fallback." >&2; exit 2 ;; \
    esac

# Where the image bakes itself.  IMAGE_ROOT is FIDELITY_ENGINE_ROOT: the bootstrap
# builds its venv at $ROOT/venv and clones the pipeline at $ROOT/pipeline, so
# pointing that root at an image directory is what puts them in a layer.
# FIDELITY_ENGINE_ROOT only: the pre-2026-08-31 spelling FIDELITY_K6_ROOT is
# still accepted as a fallback by the stage scripts, but an image and the stage
# scripts inside it ship together, so a NEW surface has no reason to be born
# carrying a deprecated name it would then have to be migrated off.
ENV FIDELITY_IMAGE_ROOT=/opt/fidelity \
    FIDELITY_ENGINE_ROOT=/opt/fidelity \
    FIDELITY_SUITE_ROOT=/opt/fidelity/suite \
    QP_PIPELINE_ROOT=/opt/fidelity/pipeline \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# python3.12 is not optional: the bootstrap refuses any other minor version,
# because the proven env recipe -- the one that produced every sealed streaming
# row -- is py3.12.  `python3` is listed SEPARATELY and on purpose: every stage
# reads job.json through `stage_measure.sh`'s jqget, which runs the SYSTEM
# python3 rather than the venv's, because it must work in `setup` -- before the
# venv exists.  On 24.04 they are the same interpreter, and relying on that
# coincidence is how the next base image breaks every stage at line one.
# git/patch are for the pipeline pin and its series; wget and ca-certificates
# are what the bootstrap's own fetches use.
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
        python3 python3.12 python3.12-venv python3.12-dev \
        build-essential git patch wget curl ca-certificates bash \
        pciutils tzdata \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version'

# The suite, staged whole and then pruned to exactly bin/BUNDLE.txt -- the one
# list that already defines "what lands on rented hardware", so the image is a
# second transport for the same audited set rather than a parallel one.
COPY bin/       /tmp/stage/bin/
COPY engines/tools/  /tmp/stage/engines/tools/
COPY engines/coverage.json /tmp/stage/engines/
COPY engines/quant-coverage-audit.json /tmp/stage/engines/
COPY engines/patches-v2/ /tmp/stage/engines/patches-v2/
# Committed panels: a container-native root capture reads its panel from
# the image itself (BUNDLE.txt lists the files; --require-all makes a
# missing one a BUILD error, not a mid-rental one).
COPY engines/panels/ /tmp/stage/engines/panels/
# Committed scopes: the on-box gguf selftest re-authors one from the bundled
# evidence and compares against the committed file (BUNDLE.txt names it).
COPY engines/scopes/ /tmp/stage/engines/scopes/
COPY registry/  /tmp/stage/registry/
# docs/schema/ is DATA, not documentation: `fidelity_dataset.py capture` ends in
# a postcondition that validates the manifest it just wrote, and dsvalidate
# os.listdirs this directory to build its schema registry. Leaving it out does
# not fail the build -- it ships an image that dies at the end of the capture
# stage, after the bootstrap, the fetch and the capture are all paid for.
# --require-all below is what turns that class of omission into a build error.
COPY docs/schema/ /tmp/stage/docs/schema/
RUN python3.12 /tmp/stage/bin/container_prune.py \
        --stage /tmp/stage --out /opt/fidelity/suite --require-all \
 && rm -rf /tmp/stage

# The patch series has to sit at $ROOT/patches-v2, which is where the runtime
# `setup` stage copies it too; doing it here as well is what lets the bootstrap
# apply the series at BUILD time instead of on every rental.
RUN mkdir -p /opt/fidelity/patches-v2 \
 && cp -f /opt/fidelity/suite/engines/patches-v2/* /opt/fidelity/patches-v2/

# Steps 1-4 of the bootstrap: python, the pinned wheel set, the pipeline at its
# pin with the measurement patch series, and the exllamav3 decision (which the
# probe makes, not an assumption).  Steps 5 and 6 are pre-flight CHECKS and one
# of them re-decodes on THIS BOX'S CUDA device, which a builder does not have;
# they run at container start, where they mean something.
RUN FIDELITY_FS_ROOT=/opt/fidelity/suite \
    FIDELITY_BOOTSTRAP_INSTALL_ONLY=1 \
    bash /opt/fidelity/suite/bin/bootstrap_measure.sh

# What this image IS, recorded from what the build actually resolved rather
# than from what it asked for.  image_content_sha256 is the digest of that
# manifest: a registry digest is stripped by `docker load`, so a content pin is
# the only identity that survives every transport.
RUN python3.12 /opt/fidelity/suite/bin/container_manifest.py \
        --image-root /opt/fidelity --suite /opt/fidelity/suite \
        --suite-revision "$SUITE_REVISION" \
        --image-reference "$IMAGE_REFERENCE"

# The run root is a MOUNT, always.  Defaulting it into the image would put a
# 200 GB fetch on the container's own layer, which is the shape of the bug that
# made a restarted pod skip `setup` as done while the tree it wrote had
# evaporated.
# OCI labels: what a puller can read without running anything.  revision is the
# same value the receipt's produced_by will carry.
LABEL org.opencontainers.image.title="quant-fidelity-measure" \
      org.opencontainers.image.description="Receipt-backed quantization fidelity measurement: the stage sequence, the pinned stack, and nothing else." \
      org.opencontainers.image.source="https://github.com/malaiwah/quant-fidelity-suite" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${SUITE_REVISION}" \
      org.opencontainers.image.ref.name="${IMAGE_REFERENCE}"

VOLUME ["/workspace"]
ENV FIDELITY_FS_ROOT=/workspace/fidelity
WORKDIR /workspace

ENTRYPOINT ["python3.12", "/opt/fidelity/suite/bin/container_entry.py"]
CMD ["doctor"]

# HF Jobs consumes a sealed workflow plan, not container_entry's local stage
# plan. Keep the base BUILD.json truthful; JOBS.json identifies only this overlay.
FROM ${HF_JOBS_BASE_IMAGE} AS hf-jobs
ARG HF_JOBS_BASE_IMAGE
ARG SUITE_REVISION=""
ARG IMAGE_REFERENCE=""
# Installed exactly like the ssh stage's entrypoint: COPY plus an explicit
# chmod, so the classic builder can produce this overlay without BuildKit.
# hf_job_manifest.py refuses anything that is not an installed 0755 regular
# file, so the mode is verified rather than assumed either way.
COPY bin/hf_job_entry.py /usr/local/bin/qfs-job
RUN chmod 0755 /usr/local/bin/qfs-job
COPY bin/hf_job_manifest.py /tmp/hf_job_manifest.py
RUN /opt/fidelity/venv/bin/python /tmp/hf_job_manifest.py \
        --suite-revision "$SUITE_REVISION" \
        --base-image "$HF_JOBS_BASE_IMAGE" \
 && rm /tmp/hf_job_manifest.py
LABEL org.opencontainers.image.title="quant-fidelity HF Jobs launcher" \
      org.opencontainers.image.description="Sealed capture, measure and compare CLI over the unchanged reviewed measurement runtime." \
      org.opencontainers.image.revision="${SUITE_REVISION}" \
      org.opencontainers.image.ref.name="${IMAGE_REFERENCE}"
ENTRYPOINT ["/usr/local/bin/qfs-job"]
CMD ["--help"]

# ---------------------------------------------------------------------------
# `ssh`: the same image, driven by the safe RunPod SSH controller.
#
# measure-cloud's RunPod path is an SSH transport (host key authenticated from
# the provider's log stream, exact retrieval, live attestation); until now it
# ran on runpod/pytorch and rebuilt the locked stack on every pod.  This
# target adds sshd and RunPod's PUBLIC_KEY convention on top of the baked
# stack (host keys are generated at boot, never baked: a key shared by every
# pod would let anyone holding the image impersonate a pod to the
# controller), so a pod boots with the pinned closure at /opt/fidelity and the
# bootstrap seeds the per-attempt engine root from it instead of
# downloading it (bin/bootstrap_measure.sh, seed_from_image).  The bundle the
# controller uploads stays the code identity the receipts bind to.
#
#   container/build.sh --target ssh --tag ghcr.io/malaiwah/quant-fidelity-measure:ssh
# ---------------------------------------------------------------------------
FROM base AS ssh
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends openssh-server \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd /root/.ssh \
 && chmod 0700 /root/.ssh \
 && rm -f /etc/ssh/ssh_host_*
COPY container/ssh-start.sh /opt/fidelity/ssh-start.sh
RUN chmod 0755 /opt/fidelity/ssh-start.sh
EXPOSE 22
ENTRYPOINT ["/opt/fidelity/ssh-start.sh"]
CMD []

# The default target stays the entrypoint-driven measurement image.
FROM base AS measure
