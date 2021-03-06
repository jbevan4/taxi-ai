.PHONY:
	play_taxi

play_taxi:
	docker build -t taxi .
	docker run -it --rm taxi