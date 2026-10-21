"""Save GRPO metrics as paired step/time plots, with means and std across seeds."""

import argparse
import json
from pathlib import Path
import statistics


METRICS = {
    "train/loss": "Policy-gradient loss",
    "train/grad_norm_before_clip": "Gradient norm before clipping",
    "train/token_entropy": "Response-token entropy (nats)",
    "reward/mean_reward": "Training total reward",
    "reward/mean_format_reward": "Training format reward",
    "val/mean_reward": "Validation total reward / accuracy",
    "val/mean_format_reward": "Validation format reward",
    "val/mean_response_tokens": "Validation response length (tokens)",
}

COMPARISONS = {
    "reward_comparison": ("Total reward / strict accuracy", "reward/mean_reward", "val/mean_reward"),
    "format_comparison": ("Format compliance", "reward/mean_format_reward", "val/mean_format_reward"),
    "length_comparison": ("Response length (tokens)", "rollout/mean_response_tokens", "val/mean_response_tokens"),
}
ELAPSED = "time/elapsed_seconds"
OPTIONAL_METRICS = {
    "policy/clip_fraction": "Clip fraction (forwarded tokens or sequences)",
    "train/fraction_pruned_sequences": "Fraction of zero-advantage sequences pruned",
    "progress/policy_lag_updates": "Policy age before update (optimizer steps)",
}


def load_runs(directories):
    runs = []
    comparable = (
        "model", "prompt_template", "reward_function", "train_data_sha256", "val_data_sha256",
        "n_train_examples", "n_val_examples", "num_rollout_steps", "learning_rate", "group_size",
        "rollout_batch_size", "gradient_accumulation_steps", "sampling_max_tokens",
        "sampling_temperature", "sampling_top_p", "max_grad_norm", "advantage_eps", "eval_seed",
        "eval_every", "eval_batch_size", "algorithm",
    )
    for directory in directories:
        config = json.loads((directory / "config.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if status["status"] != "complete":
            raise ValueError(f"Run is not complete: {directory}")
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line]
        if runs and any(config[key] != runs[0]["config"][key] for key in comparable):
            raise ValueError("Only combine runs with the same experiment settings and datasets")
        if runs:
            def off_policy_settings(settings):
                return (settings.get("experiment", "standard_on_policy"),
                        settings.get("train_batch_size", settings["rollout_batch_size"]),
                        settings.get("cliprange"), settings.get("log_probs_dtype", "model"))
            if off_policy_settings(config) != off_policy_settings(runs[0]["config"]):
                raise ValueError("Only combine runs with the same off-policy settings and log-probability precision")
        if any(config["seed"] == run["config"]["seed"] for run in runs):
            raise ValueError("Each run must have a distinct seed")
        runs.append({"config": config, "rows": rows})
    return runs


def metric_series(runs, metric):
    by_run = [{row["step"]: row[metric] for row in run["rows"] if metric in row} for run in runs]
    if not by_run or not by_run[0] or any(series.keys() != by_run[0].keys() for series in by_run):
        raise ValueError(f"Missing or inconsistent step coverage for {metric}")
    steps = sorted(by_run[0])
    values = [[series[step] for step in steps] for series in by_run]
    means = [statistics.mean(column) for column in zip(*values)]
    stds = [statistics.stdev(column) if len(column) > 1 else 0.0 for column in zip(*values)]
    return steps, values, means, stds


def plot_series(runs, metric, wall_time=False, step_axis="optimizer"):
    """Return per-seed x/y and aggregate x/mean/std, aligned by optimizer update.

    On the time axis, aggregate x is mean elapsed time at the matching update.
    This is deliberately not an interpolation/aggregation at fixed elapsed time.
    Untimed updates are omitted, never assigned an invented timestamp.
    """
    if step_axis == "rollout":
        # Keep completed rollout batches only, including the initial evaluation.
        # Training metrics are from the final update, not a rollout-wide average.
        runs = [{**run, "rows": [
            {**row, "step": row.get("progress/rollout_step", row["step"])}
            for row in run["rows"]
            if row.get("progress/update_in_rollout", 1) == row.get("progress/updates_per_rollout", 1)
        ]} for run in runs]
    steps, values, means, stds = metric_series(runs, metric)
    if not wall_time:
        return [steps for _ in runs], values, steps, means, stds
    times = [{row["step"]: row[ELAPSED] for row in run["rows"] if ELAPSED in row} for run in runs]
    indices = [i for i, step in enumerate(steps) if all(step in series for series in times)]
    x_values = [[series[steps[i]] for i in indices] for series in times]
    y_values = [[sequence[i] for i in indices] for sequence in values]
    mean_times = [statistics.mean(column) for column in zip(*x_values)]
    return x_values, y_values, mean_times, [means[i] for i in indices], [stds[i] for i in indices]


def draw_metric(ax, runs, metric, wall_time, label, color, linestyle, step_axis="optimizer"):
    x_values, values, mean_x, means, stds = plot_series(runs, metric, wall_time, step_axis)
    if not mean_x:
        ax.text(0.5, 0.5, "Elapsed time unavailable", ha="center", va="center", transform=ax.transAxes)
        return
    if len(runs) > 1:
        for xs, sequence in zip(x_values, values, strict=True):
            ax.plot(xs, sequence, color=color, alpha=0.22, linewidth=0.8, linestyle=linestyle)
        ax.fill_between(mean_x, [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)], color=color, alpha=0.14)
        label += " mean"
    ax.plot(mean_x, means, color=color, linewidth=1.8, linestyle=linestyle,
            marker="o", markersize=3.5, label=label)


def draw_pair(axes, runs, title, curves, step_axis="optimizer"):
    """Draw the same metric(s) against completed updates and measured elapsed time."""
    for ax, wall_time in zip(axes, (False, True), strict=True):
        for metric, label, color, linestyle in curves:
            draw_metric(ax, runs, metric, wall_time, label, color, linestyle, step_axis)
        step_label = "Rollout batches (completed)" if step_axis == "rollout" else "Gradient steps (completed updates)"
        ax.set(xlabel="Wall-clock time (seconds)" if wall_time else step_label,
               ylabel=title, title="By elapsed time" if wall_time else f"By {step_axis} step")
        ax.set_xlim(left=0)
        if all(metric.endswith(("mean_reward", "mean_format_reward")) for metric, *_ in curves):
            ax.set_ylim(-0.025, 1.025)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=9)


def figure_notes(runs):
    notes = ["Time includes rollouts, optimizer updates and validation; excludes model/server startup."]
    if any(ELAPSED not in row for run in runs for row in run["rows"]):
        notes.append("Points without timestamps appear only on the step axis; no timestamps are inferred.")
    if len(runs) > 1:
        notes.append("Faint lines: seeds; bands: ±1 sample std. Time-axis means/bands align matching updates, using mean elapsed time.")
    return "\n".join(notes)


def save_figure(fig, output_dir, name, title, notes, dpi):
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015, notes, ha="center", va="bottom", fontsize=8, color="#555555")
    height = fig.get_size_inches()[1]
    # tight_layout reserves the suptitle; reserve extra space for the footer.
    bottom = (0.35 + 0.14 * len(notes.splitlines())) / height
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(output_dir / f"{name}.png", dpi=dpi)
    fig.savefig(output_dir / f"{name}.pdf")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Per-seed output directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", help="Experiment label (defaults to the first run's parent directory)")
    parser.add_argument("--dpi", type=int, default=200, help="PNG resolution; PDF remains vector graphics")
    parser.add_argument("--step-axis", choices=("optimizer", "rollout"), default="optimizer",
                        help="Rollout mode uses only the final update in each rollout batch")
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    runs = load_runs(args.runs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "figure.facecolor": "white", "axes.axisbelow": True})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = runs[0]["config"]
    name = args.title or args.runs[0].parent.name
    seeds = ", ".join(str(run["config"]["seed"]) for run in runs)
    subtitle = f"{config['model'].rsplit('/', 1)[-1]} | lr={config['learning_rate']:g} | group={config['group_size']} | seeds: {seeds}"
    notes = figure_notes(runs)
    if args.step_axis == "rollout":
        notes += "\nCompleted rollout batches only; training metrics show each batch's final update."
    if any(any("policy/clip_fraction" in row for row in run["rows"]) for run in runs):
        notes += "\nClipping/entropy describe forwarded responses; zero coverage is logged as zero."
    summaries = {}
    metrics = METRICS | {metric: label for metric, label in OPTIONAL_METRICS.items()
                         if all(any(metric in row for row in run["rows"]) for run in runs)}
    for metric, label in metrics.items():
        steps, values, means, stds = metric_series(runs, metric)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        draw_pair(axes, runs, label, [(metric, label, "#1f77b4", "-")], args.step_axis)
        filename = metric.replace("/", "_")
        save_figure(fig, args.output_dir, filename, f"{name} — {label}\n{subtitle}", notes, args.dpi)
        plt.close(fig)
        summaries[metric] = {"final_step": steps[-1], "mean": means[-1], "sample_std": stds[-1],
                             "per_seed": {str(run["config"]["seed"]): sequence[-1] for run, sequence in zip(runs, values)}}

    comparisons = {}
    for filename, (label, train_metric, val_metric) in COMPARISONS.items():
        # Older logs may lack the optional rollout length metric.
        if not all(any(train_metric in row for row in run["rows"]) for run in runs):
            continue
        curves = [(train_metric, "Train rollouts", "#1f77b4", "-"), (val_metric, "Validation", "#ff7f0e", "--")]
        comparisons[filename] = (label, curves)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
        draw_pair(axes, runs, label, curves, args.step_axis)
        save_figure(fig, args.output_dir, filename, f"{name} — {label}\n{subtitle}", notes, args.dpi)
        plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharey="row")
    overview_rows = [comparisons["reward_comparison"], comparisons["format_comparison"],
                     (METRICS["train/token_entropy"], [("train/token_entropy", "Train", "#1f77b4", "-")])]
    for pair, (label, curves) in zip(axes, overview_rows, strict=True):
        draw_pair(pair, runs, label, curves, args.step_axis)
    save_figure(fig, args.output_dir, "overview", f"{name} — GRPO learning curves\n{subtitle}", notes, args.dpi)
    plt.close(fig)

    result = {"num_seeds": len(runs), "metrics": summaries, "step_axis": args.step_axis,
              "wall_time_aggregation": "Mean elapsed time at matching updates; std across seeds at matching updates.",
              "untimed_points": [{"seed": run["config"]["seed"], "step": row["step"]}
                                 for run in runs for row in run["rows"] if ELAPSED not in row]}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    accuracy = summaries["val/mean_reward"]["mean"]
    seed_label = "seed" if len(runs) == 1 else "seeds"
    print(f"{len(runs)} {seed_label}; final mean validation reward: {accuracy:.2%}. Plots: {args.output_dir}")


if __name__ == "__main__":
    main()
