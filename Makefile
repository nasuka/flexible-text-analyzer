.PHONY: help up down build logs lint format check

help:
	@echo "Available commands:"
	@echo "  up      - Start the Docker containers"
	@echo "  down    - Stop the Docker containers"
	@echo "  build   - Build the Docker image"
	@echo "  logs    - Show container logs"
	@echo "  lint    - Run ruff linting"
	@echo "  format  - Run ruff formatting"
	@echo "  check   - Run both lint and format"

up:
	docker-compose up --build

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

lint:
	ruff check .

format:
	ruff format .

check: lint format
	@echo "Linting and formatting completed"