# Convenience wrapper around docker compose for the sentry nav sim rig.
# Run from this directory: `make build`, `make sim`, `make brain`, etc.
# `make df` first — the images build to several GB (see README "Disk").

COMPOSE := docker compose

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: df
df: ## Show free disk before building (need a few GB headroom)
	@df -h / | tail -2
	@docker system df

.PHONY: build
build: ## Build both images
	$(COMPOSE) build

.PHONY: build-nav2
build-nav2: ## Build only the nav2 (ROS+Nav2+Gazebo) image
	$(COMPOSE) build nav2

.PHONY: build-autox
build-autox: ## Build only the autox image
	$(COMPOSE) build autox

.PHONY: sim
sim: ## Rig 1: Gazebo full sim (odom+scan+cmd_vel from Gazebo)
	$(COMPOSE) run --rm nav2 ros2 launch sam_bot_bringup bringup.launch.py

.PHONY: brain
brain: ## Run AutoX AutoNav/brain against the running stack (ROS=NAV2 MCU=MOCK)
	$(COMPOSE) run --rm autox python main.py @MOCK_CAM @SENTRY ROS=NAV2 MCU=MOCK

.PHONY: rig2
rig2: ## Rig 2: MCU bridge <-> mock kinematic odom (/odom advances under /cmd_vel)
	$(COMPOSE) run --rm autox bash /autox/docker/rig2_mcu_bridge.sh

.PHONY: bench-mcu
bench-mcu: ## Latency: AutoX in-process McuDriver round-trip ('t'/'q', isolated + loaded)
	$(COMPOSE) run --rm autox python docker/bench_mcu_latency.py

.PHONY: bench-ros
bench-ros: ## Latency: ROS/DDS service round-trip (the cost kept off the aim path)
	$(COMPOSE) run --rm autox python docker/bench_ros_latency.py

.PHONY: evade
evade: ## Evade test: AutoNav + fake auto-aim vs a running sim (`make sim` first)
	$(COMPOSE) run --rm autox python run_evade_test.py autonav:nav2_localizer:none

.PHONY: teleop
teleop: ## Drive /cmd_vel by keyboard (rig 2 manual check)
	$(COMPOSE) run --rm nav2 ros2 run teleop_twist_keyboard teleop_twist_keyboard

.PHONY: topics
topics: ## List the live ROS graph from inside the nav2 container
	$(COMPOSE) run --rm nav2 ros2 topic list

.PHONY: shell-nav2
shell-nav2: ## Shell into the nav2 container (workspace built + sourced)
	$(COMPOSE) run --rm nav2 bash

.PHONY: shell-autox
shell-autox: ## Shell into the autox container (venv synced + ROS sourced)
	$(COMPOSE) run --rm autox bash

.PHONY: config
config: ## Validate + render the merged compose config
	$(COMPOSE) config

.PHONY: down
down: ## Stop and remove any running rig containers
	$(COMPOSE) down --remove-orphans

.PHONY: clean
clean: ## Remove the rig images (reclaim disk)
	-docker image rm sentry-nav-rig/nav2:humble sentry-nav-rig/autox:humble
	docker builder prune -f
