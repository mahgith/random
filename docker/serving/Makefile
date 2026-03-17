PROJECT_ID       ?= your-gcp-project-id
REGION           ?= europe-west1
REPO             ?= ml-images
FORECASTING_TAG  ?= latest
IMAGE_NAME       := serving
VERSION          := $(shell cat docker/serving/version.txt 2>/dev/null || echo latest)
IMAGE_URI        := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/$(IMAGE_NAME):$(VERSION)
FORECASTING_URI  := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/forecasting:$(FORECASTING_TAG)

.PHONY: build push build-push

build:
	docker build -f docker/serving/Dockerfile \
	  --build-arg BASE_IMAGE_URI=$(FORECASTING_URI) \
	  -t $(IMAGE_URI) .

push:
	docker push $(IMAGE_URI)

build-push: build push
