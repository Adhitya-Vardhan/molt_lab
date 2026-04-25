"""FastAPI app for MolForge."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run MolForge. Install dependencies from pyproject.toml."
    ) from exc

try:
    from ..models import MolForgeAction, MolForgeObservation
    from .molforge_environment import MolForgeEnvironment
except ImportError:
    from models import MolForgeAction, MolForgeObservation
    from server.molforge_environment import MolForgeEnvironment


app = create_app(
    MolForgeEnvironment,
    MolForgeAction,
    MolForgeObservation,
    env_name="molforge",
    max_concurrent_envs=2,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the environment locally without Docker."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
