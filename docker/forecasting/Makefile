PROJECT_ID  ?= your-gcp-project-id
REGION      ?= europe-west1
REPO        ?= ml-images
BASE_TAG    ?= latest
IMAGE_NAME  := forecasting
VERSION     := $(shell cat docker/forecasting/version.txt 2>/dev/null || echo latest)
IMAGE_URI   := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(IMAGE_NAME):$(VERSION)
BASE_URI    := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/base:$(BASE_TAG)

.PHONY: build push build-push

build:
	docker build -f docker/forecasting/Dockerfile \
	  --build-arg BASE_IMAGE_URI=$(BASE_URI) \
	  -t $(IMAGE_URI) .

push:
	docker push $(IMAGE_URI)

build-push: build push
