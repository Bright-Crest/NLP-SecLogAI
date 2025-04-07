include .env

ifeq ($(USE_GPU), true)
	TAG = latest
else
	TAG = cpu
endif

IMAGE_NAME = nlp-seclogai
WEB_APP_CONTAINER_NAME = $(IMAGE_NAME)-web-app

# 构建镜像
build:
	docker build -t $(IMAGE_NAME):$(TAG) . --build-arg USE_GPU=$(USE_GPU) --build-arg CUDA=$(CUDA)

# 运行容器
run: 
	docker run --name $(WEB_APP_CONTAINER_NAME) -d -p 5000:5000 --env-file .env --gpus all \
 		$(IMAGE_NAME):$(TAG)

# 停止容器
stop:
	docker stop $(WEB_APP_CONTAINER_NAME)

# 删除容器
rm: 
	docker rm $(WEB_APP_CONTAINER_NAME)

# 进入容器
bash:
	docker run --name $(IMAGE_NAME)-bash -it --gpus all \
		$(IMAGE_NAME):$(TAG) \
		/bin/bash

# 测试训练
test-train:
	if docker ps -a | grep $(IMAGE_NAME)-test-train; then \
		docker rm $(IMAGE_NAME)-test-train; \
	fi
	docker run --name $(IMAGE_NAME)-test-train -it --gpus all \ 
		$(IMAGE_NAME):$(TAG) \
		python ai_detect/train.py \
		--train_file ai_detect/dataset/Linux/Linux.log \
		--num_epochs 1 \
		--not_use_tensorboard

# 测试评估
test-eval:
	if docker ps -a | grep $(IMAGE_NAME)-test-eval; then \
		docker rm $(IMAGE_NAME)-test-eval; \
	fi
	docker run --name $(IMAGE_NAME)-test-eval -it --gpus all \
		$(IMAGE_NAME):$(TAG) \
		python ai_detect/evaluate.py \
		--test_file ai_detect/dataset/Linux/Linux.log \

# 测试
test:
	docker run --name $(IMAGE_NAME)-test -it --rm --gpus all \
		$(IMAGE_NAME):$(TAG) \
		pytest tests/
