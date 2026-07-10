ADK ?= uv run adk
ADK_HOST ?= 127.0.0.1
ADK_PORT ?= 8000
ADK_AGENTS_DIR ?= src/visionops_crew/agents
ADK_PATTERN = [a]dk web --reload_agents
JUPYTER_HOST ?= 127.0.0.1
JUPYTER_PORT ?= 4040
JUPYTER_ROOT ?= .

.PHONY: run-mongo mongo-down mongo-restart mongo-reset-data mongo-reset mongo-logs run-crew stop-crew run-jupyter

run-mongo:
	docker compose up -d --wait mongo

mongo-down:
	docker compose down

mongo-restart:
	docker compose down
	docker compose up -d --wait mongo

mongo-reset-data:
	docker compose down
	rm -rf docker-data/mongo

mongo-reset:
	docker compose down
	rm -rf docker-data/mongo
	docker compose up -d --wait mongo

mongo-logs:
	docker compose logs -f mongo

stop-crew:
	@pkill -f "$(ADK_PATTERN)" 2>/dev/null || true
	-@pids="$$(lsof -tiTCP:$(ADK_PORT) -sTCP:LISTEN 2>/dev/null)"; \
	if [ -n "$$pids" ]; then \
		kill $$pids 2>/dev/null || true; \
		sleep 1; \
		pids="$$(lsof -tiTCP:$(ADK_PORT) -sTCP:LISTEN 2>/dev/null)"; \
		if [ -n "$$pids" ]; then \
			kill -9 $$pids 2>/dev/null || true; \
		fi; \
	fi

run-crew: stop-crew
	$(ADK) web --reload_agents --host $(ADK_HOST) --port $(ADK_PORT) $(ADK_AGENTS_DIR)

run-jupyter:
	@if [ ! -f .env ]; then \
		echo ".env not found. Define JUPYTER_TOKEN in .env first."; \
		exit 1; \
	fi; \
	set -a; . ./.env; set +a; \
	if [ -z "$$JUPYTER_TOKEN" ]; then \
		echo "JUPYTER_TOKEN is not set in .env"; \
		exit 1; \
	fi; \
	port="$(JUPYTER_PORT)"; \
	if [ -n "$$JUPYTER_URL" ]; then \
		parsed_port="$${JUPYTER_URL##*:}"; \
		parsed_port="$${parsed_port%%/*}"; \
		if [ -n "$$parsed_port" ] && [ "$$parsed_port" != "$$JUPYTER_URL" ]; then \
			port="$$parsed_port"; \
		fi; \
	fi; \
	uv run jupyter lab \
		--ServerApp.ip="$(JUPYTER_HOST)" \
		--ServerApp.port="$$port" \
		--IdentityProvider.token="$$JUPYTER_TOKEN" \
		--ServerApp.open_browser=False \
		--ServerApp.root_dir="$(JUPYTER_ROOT)"
