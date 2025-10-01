"""Builds the project status table in `profile/project_status.md`."""

HEADER = """# Project status

| name | pypi | actions | pulls |
| ---- | ---- | ------- | ----- |
"""


PROJECTS = [
    # project_name, repo_link, pypi_exists, [options]
    ("agjax", "https://github.com/mfschubert/agjax"),
    ("ccmaps", "https://github.com/mfschubert/ccmaps"),
    ("fmmax", "https://github.com/mfschubert/fmmax"),
    ("jeig", "https://github.com/mfschubert/jeig"),
    ("mewtax", "https://github.com/mfschubert/mewtax"),
    ("refractiveindex2", "https://github.com/mfschubert/refractiveindex2"),
    ("sparsejac", "https://github.com/mfschubert/sparsejac"),
    ("invrs-gym", "https://github.com/invrs-io/gym"),
    ("invrs-opt", "https://github.com/invrs-io/opt"),
    ("invrs-utils", "https://github.com/invrs-io/utils"),
    ("leaderboard", "https://github.com/invrs-io/leaderboard", {"pypi_exists": False}),
    ("totypes", "https://github.com/invrs-io/totypes"),
    ("imageruler", "https://github.com/nanocomp/imageruler"),
    (
        "photonics-opt-testbed",
        "https://github.com/nanocomp/photonics-opt-testbed",
        {"pypi_exists": False, "actions_job": "measure_length_scales.yml"},
    ),
]

DEFAULT_ACTIONS_JOB = "build-ci.yml"


def project_line(
    project_name: str,
    repo_link: str,
    options={},
) -> str:
    """Generates the entry in the project status table for a single repo."""
    repo_owner, repo_name = repo_link.split("/")[-2:]

    name_col = f"[{project_name}]({repo_link})"

    if "pypi_exists" in options and options["pypi_exists"] is False:
        pypi_col = "n/a"
    else:
        pypi_col = f"[![pypi](https://img.shields.io/pypi/v/{project_name})](https://pypi.org/project/{project_name}/)"

    if "actions_job" in options:
        actions_job = options["actions_job"]
    else:
        actions_job = DEFAULT_ACTIONS_JOB
    actions_col = f"[![actions]({repo_link}/actions/workflows/{actions_job}/badge.svg)]({repo_link}/actions)"

    pulls_col = f"[![actions](https://img.shields.io/github/issues-pr/{repo_owner}/{repo_name})]({repo_link}/pulls)"

    return f"| {name_col} | {pypi_col} | {actions_col} | {pulls_col} |\n"


if __name__ == "__main__":

    with open("profile/project_status.md", "w") as f:
        f.write(HEADER)
        for project_data in PROJECTS:
            f.write(project_line(*project_data))
