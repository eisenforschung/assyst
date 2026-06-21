import os

from hypothesis import settings
from hypothesis.database import (
    DirectoryBasedExampleDatabase,
    GitHubArtifactDatabase,
    MultiplexedDatabase,
    ReadOnlyDatabase,
)


if os.environ.get("GITHUB_ACTIONS") == "true":
    _local = DirectoryBasedExampleDatabase(".hypothesis/examples")
    _shared = ReadOnlyDatabase(
        GitHubArtifactDatabase(
            "eisenforschung",
            "assyst",
            artifact_name=os.environ.get(
                "HYPOTHESIS_ARTIFACT_NAME", "hypothesis-example-db"
            ),
        )
    )
    settings.register_profile("ci", database=MultiplexedDatabase(_local, _shared))
    settings.load_profile("ci")
