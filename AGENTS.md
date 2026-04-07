# Repository Guidelines

## Project Structure & Module Organization
This repository is split into two Python packages. `legged_gym/legged_gym/` contains simulation environments, task configs, utilities, and executable scripts. Key areas are `envs/` for robot/task definitions, `scripts/` for train/play/export entry points, `utils/` for argument parsing and task registration, and `tests/` for smoke tests. `rsl_rl/rsl_rl/` contains the PPO runner, storage, and policy modules used by the environments. Robot URDFs, meshes, and actuator assets live under `legged_gym/resources/`. Top-level `images/` stores documentation assets.

## Build, Test, and Development Commands
Use the Python 3.8 environment described in `README.md`, with Isaac Gym installed separately.

```bash
pip install -e rsl_rl
pip install -e legged_gym
pip install -r legged_gym/requirements.txt
python legged_gym/legged_gym/scripts/train.py --task a1 --exptid 001-00-baseline --device cuda:0
python legged_gym/legged_gym/scripts/play.py --task a1 --exptid 001-00
python legged_gym/legged_gym/tests/test_env.py --task a1
```

`train.py` launches PPO training, `play.py` replays a checkpoint, and `test_env.py` is the quickest environment smoke test. Use `--no_wandb` for local runs if you do not want remote logging.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions, variables, and module names, and `CamelCase` for config and model classes such as `XTDogParkourCfgPPO`. Keep task-specific changes close to their config module, for example `legged_gym/legged_gym/envs/xt_dog/xt_parkour_config.py`. There is no committed formatter or linter config, so keep imports tidy, prefer small focused edits, and match the surrounding comment style.

## Testing Guidelines
There is no broad automated test suite; contributors should at minimum run the relevant smoke test and the affected script path. Add tests under `legged_gym/legged_gym/tests/` using `test_*.py` naming. For environment or reward changes, include the exact command used to validate behavior and note the task (`a1`, `go1`, or `xt_dog`) in your PR.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects such as `add depth` or `update xt_parkour_config.py`. Keep the first line under 72 characters, describe one logical change per commit, and avoid placeholder messages like `tmp`. PRs should explain the behavioral change, list validation commands, link any issue, and include screenshots or short clips when modifying viewer output, terrain behavior, or robot motion.
