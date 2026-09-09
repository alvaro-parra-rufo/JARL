"""Environment and git metadata collection for experiment runs."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "EnvInfo",
    "GitInfo",
    "RunMetadata",
    "collect_env_info",
    "collect_git_info",
    "now_iso",
    "save_lockfile",
]


@dataclass
class EnvInfo:
    """Snapshot of the runtime environment.

    Args:
        python_version: Python interpreter version.
        platform: OS and architecture string.
        hostname: Machine hostname.
        packages: Mapping of key package names to their installed versions.
    """

    python_version: str = ""
    platform: str = ""
    hostname: str = ""
    packages: dict[str, str] = field(default_factory=dict)


@dataclass
class GitInfo:
    """Snapshot of the git repository state.

    Args:
        commit: Current HEAD commit hash.
        branch: Active branch name.
        dirty: Whether the working tree has uncommitted changes.
        remote_url: URL of the origin remote (if any).
    """

    commit: str = ""
    branch: str = ""
    dirty: bool = False
    remote_url: str = ""


@dataclass
class RunMetadata:
    """Full metadata for a training run.

    Args:
        start_time: ISO-8601 timestamp when the run started.
        end_time: ISO-8601 timestamp when the run finished (empty while running).
        status: Current run status.
        transfer_group: Navix map transfer group pinned at experiment root creation.
        env: Environment snapshot.
        git: Git repository snapshot.
    """

    start_time: str = ""
    end_time: str = ""
    status: str = "running"
    transfer_group: str = ""
    env: EnvInfo = field(default_factory=EnvInfo)
    git: GitInfo = field(default_factory=GitInfo)

    def to_dict(self) -> dict:
        """Convert metadata to a plain dictionary."""
        return asdict(self)

    def save(self, path: Path) -> Path:
        """Serialize metadata to a JSON file.

        Args:
            path: Destination file path.

        Returns:
            The path written to.
        """
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> RunMetadata:
        """Deserialize metadata from a JSON file.

        Args:
            path: Source file path.

        Returns:
            Reconstructed `RunMetadata` instance.
        """
        data = json.loads(path.read_text())
        return cls(
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            status=data.get("status", ""),
            transfer_group=data.get("transfer_group", ""),
            env=EnvInfo(**data.get("env", {})),
            git=GitInfo(**data.get("git", {})),
        )


def collect_env_info() -> EnvInfo:
    """Collect current runtime environment information.

    Returns:
        Populated `EnvInfo` with Python version, platform, hostname,
        and versions of key ML packages (jax, flax, optax, orbax).
    """
    packages: dict[str, str] = {}
    for pkg in ("jax", "jaxlib", "flax", "optax", "orbax", "tensorboard"):
        try:
            import importlib.metadata

            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = "not installed"

    return EnvInfo(
        python_version=platform.python_version(),
        platform=platform.platform(),
        hostname=platform.node(),
        packages=packages,
    )


def collect_git_info() -> GitInfo:
    """Collect git repository state from the current working directory.

    Returns:
        Populated `GitInfo`. Fields default to empty strings if not in a git repo.
    """

    def _run(cmd: list[str]) -> str:
        """Run a git command and return stripped stdout, or empty on failure."""
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()  # noqa: S603
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    if not commit:
        return GitInfo()

    return GitInfo(
        commit=commit,
        branch=_run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        dirty=bool(_run(["git", "status", "--porcelain"])),
        remote_url=_run(["git", "config", "--get", "remote.origin.url"]),
    )


def save_lockfile(dest_dir: Path, lockfile: str | Path = "uv.lock") -> Path | None:
    """Copy a dependency lockfile to a destination directory for reproducibility.

    Args:
        dest_dir: Directory to copy the lockfile into.
        lockfile: Path to the source lockfile. Defaults to `uv.lock` in the cwd.

    Returns:
        Path to the copied file, or `None` if the source doesn't exist.
    """
    src = Path(lockfile)
    if not src.exists():
        return None
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return dest


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=UTC).isoformat()
