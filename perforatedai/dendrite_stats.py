"""Record dendrite-to-neuron-error correlation statistics during PAI training."""

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from perforatedai import globals_perforatedai as GPA

__all__ = ["DendriteStatisticsRecord", "DendriteStatisticsRecorder"]


@dataclass
class DendriteStatisticsRecord:
    """
    Statistics for one dendrite added to one layer.

    Attributes:
        add_event_index: Which dendrite-addition event this record belongs to,
            counting from zero.
        layer_name: The name of the parent neuron module.
        epoch_added: The value of num_epochs_run when the dendrite was added.
        best_absolute_pearson_correlation: Largest absolute Pearson correlation
            across all output nodes and candidates, bounded in [0, 1].
        mean_absolute_pearson_correlation: Mean of the same quantity over nodes.
        best_absolute_covariance: Largest absolute covariance across all output
            nodes and candidates. Unbounded. This is the scoring quantity.
        mean_absolute_covariance: Mean of the same quantity over nodes.
        mean_absolute_neuron_error_at_add: Mean absolute per-node parent error
            measured during the dendrite phase that produced this dendrite.
        mean_absolute_neuron_error_after_retraining: The same measurement taken
            during the following dendrite phase, after retraining. Stays None if
            training ended before another dendrite phase ran.
        neuron_error_reduction: The decrease in mean absolute per-node error
            between those two measurements. Positive means the error fell. Stays
            None whenever the after-retraining measurement is None.
        validation_score_before: Best neuron-phase validation score before this
            dendrite was added.
        validation_score_after: Best neuron-phase validation score after this
            dendrite was added and the network was retrained.
        validation_score_improvement: The improvement between the two, signed so
            that positive always means better, whichever direction is better.
        retained: True if this dendrite generation was kept, False if it was
            tried and discarded, None if training ended before the decision was
            made. Rows with retained False describe a generation that was rolled
            back, so their error and score changes do not measure a dendrite's
            effect and should be filtered out before analysis.
    """

    add_event_index: int
    layer_name: str
    epoch_added: int
    best_absolute_pearson_correlation: float
    mean_absolute_pearson_correlation: float
    best_absolute_covariance: float
    mean_absolute_covariance: float
    mean_absolute_neuron_error_at_add: float
    retained: bool | None = None
    mean_absolute_neuron_error_after_retraining: float | None = None
    neuron_error_reduction: float | None = None
    validation_score_before: float | None = None
    validation_score_after: float | None = None
    validation_score_improvement: float | None = None


def _summarize_layer(neuron_module: Any) -> dict[str, float] | None:
    """Collapse one layer's per-node correlation buffers to scalar summaries.

    Each correlation buffer is one-dimensional, of shape (N,), holding one
    scalar per output node. The buffers are registered lazily by
    DendriteValueTracker.setup_arrays on the first backward pass, so they may
    not exist yet. On a fresh run only dendrite_values[0] is set up, because
    filter_backward calls setup_arrays on that index specifically; candidates
    beyond the first only receive buffers when a state dict is loaded. Any
    candidate whose buffers are missing is skipped rather than raising.

    Args:
        neuron_module: A PAINeuronModule taken from the tracker's
            neuron_module_vector.

    Returns:
        A dictionary of scalar summaries for this layer, or None if no
        candidate had its correlation buffers allocated yet.
    """
    candidate_count: int = GPA.pc.get_global_candidates()
    pearson_per_candidate: list[torch.Tensor] = []
    covariance_per_candidate: list[torch.Tensor] = []

    for candidate_index in range(candidate_count):
        values = neuron_module.dendrite_module.dendrite_values[candidate_index]
        pearson = getattr(values, "best_pearson_correlation", None)
        covariance = getattr(values, "prev_dendrite_candidate_correlation", None)
        if pearson is None or covariance is None:
            continue
        pearson_per_candidate.append(pearson.detach().abs().float().cpu())
        covariance_per_candidate.append(covariance.detach().abs().float().cpu())

    if not pearson_per_candidate:
        return None

    # Dimensions: allocated_candidate_count x number_of_output_nodes.
    stacked_pearson: torch.Tensor = torch.stack(pearson_per_candidate)
    stacked_covariance: torch.Tensor = torch.stack(covariance_per_candidate)

    # The parent error is a property of the neuron, not of any one candidate,
    # so candidate zero is representative.
    parent_error_buffer = getattr(
        neuron_module.dendrite_module.dendrite_values[0],
        "normal_pass_average_d",
        None,
    )
    if parent_error_buffer is None:
        return None
    parent_error: torch.Tensor = parent_error_buffer.detach().abs().float().cpu()

    return {
        "best_absolute_pearson_correlation": stacked_pearson.max().item(),
        "mean_absolute_pearson_correlation": stacked_pearson.mean().item(),
        "best_absolute_covariance": stacked_covariance.max().item(),
        "mean_absolute_covariance": stacked_covariance.mean().item(),
        "mean_absolute_neuron_error": parent_error.mean().item(),
    }


class DendriteStatisticsRecorder:
    """Collect per-dendrite correlation and error-reduction statistics.

    The recorder must be driven from the training loop because the values it
    reads are overwritten by the mode switch inside add_validation_score. Call
    capture_before_validation immediately before that call and
    observe_after_validation immediately after it, every epoch.
    """

    def __init__(self, output_directory: Path) -> None:
        """Initialize the recorder.

        Args:
            output_directory: Directory the CSV file is written into. Created if
                it does not exist.
        """
        self._output_directory: Path = output_directory
        self._output_directory.mkdir(parents=True, exist_ok=True)
        self._records: list[DendriteStatisticsRecord] = []
        self._latest_snapshot: dict[str, dict[str, float]] = {}
        self._previous_dendrite_count: int = 0
        self._pending_records: list[DendriteStatisticsRecord] = []
        self._neuron_phase_scores: list[float] = []
        self._mode_during_epoch: str | None = None
        self._snapshot_is_fresh: bool = False
        self._previous_integrated_count: int = 0

    def capture_before_validation(self) -> None:
        """Snapshot every layer's correlation buffers before the mode can switch.

        Records the current mode unconditionally, because observe_after_validation
        needs to know which phase this epoch belonged to and cannot determine that
        afterward. Takes the buffer snapshot only during a dendrite-training
        phase, when the buffers hold meaningful values.
        """
        tracker = GPA.pai_tracker
        if not hasattr(tracker, "member_vars"):
            # The placeholder value present before model initialization.
            return

        # Read before add_validation_score gets a chance to switch it.
        self._mode_during_epoch = tracker.member_vars["mode"]

        if not GPA.pc.get_perforated_backpropagation():
            return
        if self._mode_during_epoch != "p":
            return

        snapshot: dict[str, dict[str, float]] = {}
        for neuron_module in tracker.neuron_module_vector:
            summary = _summarize_layer(neuron_module)
            if summary is not None:
                snapshot[neuron_module.name] = summary
        if not snapshot:
            # No layer has allocated its buffers yet; keep the previous snapshot.
            return
        self._latest_snapshot = snapshot
        self._snapshot_is_fresh = True

    def observe_after_validation(self, validation_score: float) -> None:
        """Detect a dendrite addition and close out the previous one.

        Args:
            validation_score: The same value passed to add_validation_score this
                epoch.
        """
        tracker = GPA.pai_tracker
        if not hasattr(tracker, "member_vars"):
            return

        # Classify the score by the mode that was active during the epoch.
        # Reading the mode now would return the post-switch value, which would
        # file a dendrite-phase score under the neuron phase at every boundary.
        if self._mode_during_epoch == "n":
            self._neuron_phase_scores.append(validation_score)

        # A rise in num_dendrites_integrated means the generation currently held
        # in _pending_records was kept. The library makes that decision late: it
        # sets should_increment_integrated at the neuron-to-dendrite switch that
        # begins the *next* generation, so the signal always arrives while the
        # previous generation's records are still open.
        integrated_count: int = tracker.member_vars.get("num_dendrites_integrated", 0)
        if integrated_count > self._previous_integrated_count:
            for record in self._pending_records:
                record.retained = True
            self._previous_integrated_count = integrated_count

        current_dendrite_count: int = tracker.member_vars["num_dendrites_added"]
        if current_dendrite_count <= self._previous_dendrite_count:
            return

        # A dendrite was added this epoch. The snapshot taken moments ago serves
        # two purposes: it is the after-retraining measurement for the previous
        # dendrite, and the at-add measurement for this one.
        self._close_pending_records(measure_error=True)

        add_event_index: int = self._previous_dendrite_count
        epoch_added: int = tracker.member_vars["num_epochs_run"]
        for layer_name, summary in self._latest_snapshot.items():
            self._pending_records.append(
                DendriteStatisticsRecord(
                    add_event_index=add_event_index,
                    layer_name=layer_name,
                    epoch_added=epoch_added,
                    best_absolute_pearson_correlation=summary[
                        "best_absolute_pearson_correlation"
                    ],
                    mean_absolute_pearson_correlation=summary[
                        "mean_absolute_pearson_correlation"
                    ],
                    best_absolute_covariance=summary["best_absolute_covariance"],
                    mean_absolute_covariance=summary["mean_absolute_covariance"],
                    mean_absolute_neuron_error_at_add=summary[
                        "mean_absolute_neuron_error"
                    ],
                    validation_score_before=self._best_neuron_phase_score(),
                )
            )

        self._neuron_phase_scores = []
        self._snapshot_is_fresh = False
        self._previous_dendrite_count = current_dendrite_count

    def _best_neuron_phase_score(self) -> float | None:
        """Return the best score from the neuron phase that just ended."""
        if not self._neuron_phase_scores:
            return None
        if GPA.pai_tracker.member_vars["maximizing_score"]:
            return max(self._neuron_phase_scores)
        return min(self._neuron_phase_scores)

    def _close_pending_records(self, measure_error: bool) -> None:
        """Fill in the after-retraining fields of the previous set of records.

        Args:
            measure_error: True only when a later dendrite phase has produced a
                fresh neuron-error measurement. When False, the error fields are
                left as None rather than being filled from the same snapshot that
                supplied the at-add value, which would fabricate a reduction of
                exactly zero.
        """
        if not self._pending_records:
            return
        if measure_error:
            # Another generation is starting and this one was never marked
            # integrated, so it was tried and discarded.
            for record in self._pending_records:
                if record.retained is None:
                    record.retained = False
        score_after: float | None = self._best_neuron_phase_score()
        maximizing: bool = GPA.pai_tracker.member_vars["maximizing_score"]

        for record in self._pending_records:
            summary = self._latest_snapshot.get(record.layer_name)
            if measure_error and self._snapshot_is_fresh and summary is not None:
                error_after: float = summary["mean_absolute_neuron_error"]
                record.mean_absolute_neuron_error_after_retraining = error_after
                record.neuron_error_reduction = (
                    record.mean_absolute_neuron_error_at_add - error_after
                )
            record.validation_score_after = score_after
            if record.validation_score_before is not None and score_after is not None:
                difference: float = score_after - record.validation_score_before
                record.validation_score_improvement = (
                    difference if maximizing else -difference
                )
            self._records.append(record)
        self._pending_records = []

    def finalize(self, file_name: str = "dendrite_statistics.csv") -> Path:
        """Close any open records and write the CSV file.

        The final dendrite's error fields are left empty, because measuring the
        error after retraining requires a subsequent dendrite phase that never
        ran. Its validation-score fields are still filled in.

        Args:
            file_name: Name of the CSV file to write.

        Returns:
            The path of the file written.
        """
        self._close_pending_records(measure_error=False)
        output_path: Path = self._output_directory / file_name
        field_names: list[str] = list(DendriteStatisticsRecord.__annotations__)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for record in self._records:
                writer.writerow(asdict(record))
        return output_path