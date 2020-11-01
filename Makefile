
build:
	docker build -t fursovia/dpeter .


up:
	docker run -itd -v ${PWD}:/notebook --name dpeter --gpus all fursovia/dpeter


exec:
	docker exec -it dpeter bash


push:
	docker push fursovia/dpeter
