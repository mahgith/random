IMAGE_NAME:= ml-training

DOCKER_REGISTRY:= europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast
REGISTRY_URL:= $(DOCKER_REGISTRY)/$(IMAGE_NAME)

BASE_IMAGE:= $(DOCKER_REGISTRY)/base:latest

# Get tags from version.yaml
VERSION_FILE:= version.yaml
TAGS:= $(shell grep "^-" $(VERSION_FILE) | sed 's/- //g')

.PHONY: build release

build:
	@echo "Building $(IMAGE_NAME) (FROM: $(BASE_IMAGE))..."
	docker buildx build --platform linux/amd64 \
		--build-arg BASE_IMAGE_URI=$(BASE_IMAGE) \
		-t $(IMAGE_NAME):local \
		-f Dockerfile ../../ \
		--load

release: build
	@echo "Publishing image..."
	@for tag in $(TAGS) ; do \
		echo "Pushing: $(REGISTRY_URL):$$tag" ; \
		docker tag $(IMAGE_NAME):local $(REGISTRY_URL):$$tag ; \
		docker push $(REGISTRY_URL):$$tag ; \
	done	