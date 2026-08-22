# Contributing to Genie Juan

Thank you for your interest in contributing to this project. Please use the workflow below when proposing changes.

## Current Asset Status

This repository has two active workflows:

1. `agent_app/` Databricks Asset Bundle for the app, ETL preparation, and Lakebase-backed runtime.
   - This is the active deployment path for the agent application.
   - Use the prep or full-deploy modes depending on whether you need ETL/bootstrap only or the full app rollout.
2. Local development in `agent_app/`.
   - Use the local bootstrap and hot-reload scripts for iterative development across the frontend, middleware, and backend.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/alexxx-db/genie-unifiedchat.git
   cd genie-unifiedchat
   ```

2. Install the app workspace dependencies:
   ```bash
   cd agent_app
   uv sync --dev
   ```

3. If your change touches Databricks integration, deployment, or app runtime, review the setup notes in `README.md` and `agent_app/tests/README.md` before testing.

4. For app-specific work, use the `agent_app/` workflow described in the repository README.

5. When you need to recreate metadata and shared infra in `dev`, use the canonical app bundle in prep mode:
   ```bash
   cd agent_app
   ./scripts/deploy.sh --target dev --prep-only
   ```

6. When you need to deploy the app stack with Lakebase, use the app bundle:
   ```bash
   cd agent_app
   ./scripts/deploy.sh --target dev --full-deploy --run
   ```

7. For local development in `agent_app`, use the local bootstrap or hot-reload scripts:
   ```bash
   cd agent_app
   ./scripts/dev-local.sh
   ./scripts/dev-local-hot-reload.sh
   ```

## Code Standards

- Follow PEP 8 conventions.
- Include type annotations for public functions.
- Keep changes focused and easy to review.
- Update documentation when behavior, configuration, or workflows change.
- Follow the formatting and line-length settings in `agent_app/pyproject.toml` where applicable.

## Linting

This repository uses `black`, `flake8`, and the app-local test runner workflow.

```bash
black agent_app etl
flake8 agent_app
```

## Testing

Run the tests that match the scope of your change:

```bash
cd agent_app
uv run pytest tests/ -v
```

Some tests require Databricks-aware configuration or access to workspace services.

## Pull Request Process

1. Open an issue first for significant changes so the approach can be discussed early.
2. Create a feature branch from `main`.
3. Make your changes with clear, descriptive commits.
4. Run the relevant lint and test commands before opening a PR.
5. Open the PR with a short summary, context for the change, and the testing you performed.
6. Address review feedback and keep the PR focused on one concern when possible.

## Security

- Never commit credentials, tokens, or other secrets.
- Let the local dev scripts (`agent_app/scripts/dev-local.sh` or
  `agent_app/scripts/dev-local-hot-reload.sh`) create and hydrate
  `agent_app/.env` from `agent_app/databricks.yml`; only add secret or
  machine-specific overrides after the initial run.
- Report security issues through coordinated disclosure. Do not use public issues or pull requests for vulnerabilities; see `SECURITY.md` and contact `security@databricks.com`.

## License

By contributing, you agree that your changes are covered by the license in `LICENSE.md`.
