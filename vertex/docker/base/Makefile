IMAGE_NAME:= base

DOCKER_REGISTRY:= europe-west9-docker.pkg.dev/forecast-sales-poc/forecast-sales-images
REGISTRY_URL:= $(DOCKER_REGISTRY)/$(IMAGE_NAME)

# Get tags from version.yaml
VERSION_FILE:= version.yaml
TAGS := $(shell grep "^-" $(VERSION_FILE) | sed 's/- //g')

.PHONY: build release

build:
	@echo "Building $(IMAGE_NAME)..."
	docker buildx build --platform linux/amd64 \
		-t $(IMAGE_NAME):local \
		-f Dockerfile ../../

release: build
	@echo "Publishing image..."
	@for tag in $(TAGS) ; do \
		echo "Pushing: $(REGISTRY_URL):$$tag" ; \
		docker tag $(IMAGE_NAME):local $(REGISTRY_URL):$$tag ; \
		docker push $(REGISTRY_URL):$$tag ; \
	done