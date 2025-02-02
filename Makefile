run:
	fastapi run main.py
dev-run:
	fastapi dev main.py

build-docker-compose:
	docker compose up -d