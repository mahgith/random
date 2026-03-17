# ── Configuration ─────────────────────────────────────────────────────────────
# Replace with your actual GCP project and Artifact Registry repo.
PROJECT_ID  ?= your-gcp-project-id
REGION      ?= europe-west1
REPO        ?= ml-images
IMAGE_NAME  := base
VERSION     := $(shell cat docker/base/version.txt 2>/dev/null || echo latest)
IMAGE_URI   := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(IMAGE_NAME):$(VERSION)

.PHONY: build push build-push

build:
	docker build -f docker/base/Dockerfile -t $(IMAGE_URI) .

push:
	docker push $(IMAGE_URI)

build-push: build push
