#!/usr/bin/env python3
import json
from typing import Any
import typer
import wandb
import db
import end_to_end

app = typer.Typer()

PROJECT = "ai2cm/microphysics-emulation"


def _get_runs(api, experiment, group, job_type):
    if experiment:
        filters = filters = {"tags": experiment}
    else:
        filters = {}

    if group:
        filters["group"] = {"$regex": group}

    if job_type:
        filters["jobType"] = {"$regex": job_type}

    return api.runs(PROJECT, filters=filters)


@app.command()
def groups(experiment: str, filter_tags: str = typer.Option("", "-f")):
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"tags": experiment})
    filter_tags = set(filter_tags.split(","))
    groups = set(run.group for run in runs if len(filter_tags & set(run.tags)) == 0)
    for group in groups:
        print(group)


def wandb2job(run: Any) -> end_to_end.ToYaml:
    if run.job_type == "prognostic_run":
        return end_to_end.PrognosticJob(
            name=run.name,
            config=run.config["config"],
            image_tag=run.config["env"]["COMMIT_SHA"],
            project="microphysics-emulation",
            bucket="vcm-ml-experiments",
        )
    if run.job_type == "train":
        config = dict(run.config)
        env = config.pop("env")
        sha = env["COMMIT_SHA"]
        return end_to_end.TrainingJob(
            name=run.name,
            config=config,
            fv3fit_image_tag=sha,
            project="microphysics-emulation",
            bucket="vcm-ml-experiments",
        )
    else:
        raise NotImplementedError(f"{run.job_type} not implemented.")


@app.command()
def reproduce(run_id: str):
    """Show the yaml required to reproduce a given wadnb run"""
    api = wandb.Api()
    run = api.run(run_id)
    job = wandb2job(run)
    print(job.to_yaml())


@app.command()
def prognostic_runs(experiment: str, filter_tags: str = typer.Option("", "-f")):
    """Show the top level metrics for prognostic runs tagged by `experiment`

    Examples:

    Rerun the piggy-backed diagnostics for all runs in an experiment::

        query-wandb.py prognostic-runs experiment/squash -f bug \
            | parallel conda run -n fv3net prognostic_run_diags piggy -s

    """
    api = wandb.Api()
    runs = api.runs(PROJECT, filters={"tags": experiment})
    db.insert_runs(runs)
    filter_tags = tuple(set(filter_tags.split(",")))
    groups_query = db.query(
        """
    SELECT group_
    FROM (
        SELECT *, max(json_each.value in (?)) as bug
        FROM runs, json_each(tags)
        WHERE job_type='prognostic_run'
        GROUP BY runs.id
    )
    WHERE not bug
    """,
        filter_tags,
    )

    # show metrics
    for group in groups_query:
        stats = query_top_level_metrics(group)
        print(json.dumps(stats))


def query_top_level_metrics(group):
    cur = db.query(
        """
        SELECT
            group_ as "group",
            max(json_extract(summary, "$.duration_seconds")) as duration_seconds,
            max(json_extract(summary, "$.global_average_cloud_5d_300mb_ppm"))
                as global_average_cloud_5d_300mb_ppm
        FROM runs
        WHERE group_ = ?
    """,
        group,
    )
    keys = [it[0] for it in cur.description]
    (row,) = cur
    return dict(zip(keys, row))


@app.command()
def runs(
    experiment: str = typer.Option("", "-e"),
    filter_tags: str = typer.Option("", "-f"),
    group: str = typer.Option("", "--group"),
    job_type: str = typer.Option("", "--job-type"),
    format: str = typer.Option("", "-o"),
):
    """
    Examples:

    Filter all runs with "bug" in tags::

        query-wandb.py runs experiment/squash -f bug \
            | jq -sr '[.[].group] | unique | .[]'
    """
    api = wandb.Api()
    runs = _get_runs(api, experiment, group, job_type)
    filter_tags = set(filter_tags.split(","))
    for run in runs:
        if len(filter_tags & set(run.tags)) == 0:
            summary = {}
            for k, v in run.summary.items():
                # ensure that summary can be serialized to json
                try:
                    json.dumps(v)
                except Exception:
                    pass
                else:
                    summary[k] = v
            if format == "json":
                d = {
                    "job_type": run.job_type,
                    "group": run.group,
                    "tags": run.tags,
                    "id": run.id,
                    "url": run.url,
                    "summary": summary,
                    "config": run.config,
                }
                print(json.dumps(d))
            else:
                print(run.group, run.job_type, run.name, run.url)


@app.command()
def tags():
    api = wandb.Api()
    runs = api.runs(PROJECT)
    tags = set.union(*(set(run.tags) for run in runs))
    for tag in tags:
        print(tag)


if __name__ == "__main__":
    app()