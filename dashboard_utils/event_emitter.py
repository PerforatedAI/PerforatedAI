from datetime import datetime

try:
    import requests
    _requests_available = True
except ModuleNotFoundError as e:
    # Only pass if the requests package itself is missing
    if e.name == "requests":
        _requests_available = False
    else:
        # requests exists but is missing a dependency
        raise

_MAX_FAILURES = 3


class DashboardEventEmitter:
    def __init__(self):
        self._failure_count = 0
        self._warned = False
        self._epochs_emitted = 0

    def reset(self):
        self._failure_count = 0
        self._warned = False
        self._epochs_emitted = 0

    def _post(self, url, payload, pc):
        if not _requests_available:
            return
        if self._failure_count >= _MAX_FAILURES:
            return
        if pc.get_dashboard_debug():
            print(f"[PAI Dashboard] Emitting event: {payload}")
        try:
            requests.post(url, json=payload, timeout=0.5)
        except Exception as e:
            self._failure_count += 1
            if not self._warned:
                print(f"[PAI Dashboard] Could not reach dashboard at {url}: {e}")
                self._warned = True
            if self._failure_count >= _MAX_FAILURES:
                print(f"[PAI Dashboard] Failed {_MAX_FAILURES} times, disabling event emission for this run.")

    def _url(self, pc):
        return pc.get_dashboard_url().rstrip("/") + "/training-events"

    def _enabled(self, pc):
        if not _requests_available:
            return False
        if not pc.get_dashboard_events_enabled():
            return False
        if self._failure_count >= _MAX_FAILURES:
            return False
        return True

    def emit_run_start(self, pc, save_name):
        self.reset()
        if not self._enabled(pc):
            return
        self._post(self._url(pc), {
            "type": "run_start",
            "model_class": save_name,
            "timestamp": datetime.now().isoformat(),
        }, pc)

    def emit_epoch(self, pc, epoch_index, phase, scores, learning_rate=None,
                   normal_time=None, pai_time=None, pb_scores=None,
                   pb_scores_current=None):
        if not self._enabled(pc):
            return
        # true_epoch: emitter-side monotonic counter, never decremented, even
        # across a rollback that steps epoch_index backward.
        self._epochs_emitted += 1
        payload = {
            "type": "epoch",
            "epoch_index": epoch_index,
            "true_epoch": self._epochs_emitted,
            "phase": phase,
            "scores": scores,
            "learning_rate": learning_rate,
            "normal_time": normal_time,
            "pai_time": pai_time,
        }
        # Omit pb_scores entirely outside dendrite scoring phases so the
        # dashboard draws a break in the line rather than a carried-forward value
        if pb_scores:
            payload["pb_scores"] = pb_scores
        if pb_scores_current:
            payload["pb_scores_current"] = pb_scores_current
        self._post(self._url(pc), payload, pc)

    def emit_switch(self, pc, switch_ordinal, epoch_index, param_count, switch_type=None):
        if not self._enabled(pc):
            return
        payload = {
            "type": "switch",
            "switch_ordinal": switch_ordinal,
            "epoch_index": epoch_index,
            "true_epoch": self._epochs_emitted,
            "param_count": param_count,
        }
        # switch_type names the phase being entered, not the one that ended.
        # Omit rather than guess: the dashboard renders a missing value as
        # unknown, but would draw "n" as a confident claim
        if switch_type in ("n", "p"):
            payload["switch_type"] = switch_type
        self._post(self._url(pc), payload, pc)

    def emit_dendrite_added(self, pc, epoch_index, num_dendrites_integrated, param_count):
        if not self._enabled(pc):
            return
        self._post(self._url(pc), {
            "type": "dendrite_added",
            "epoch_index": epoch_index,
            "true_epoch": self._epochs_emitted,
            "num_dendrites_integrated": num_dendrites_integrated,
            "param_count": param_count,
        }, pc)

    def emit_run_end(self, pc, epoch_last_improved, global_best_score):
        if not self._enabled(pc):
            return
        # Send PerforatedAI's own stored "best" values verbatim - no argmax over
        # accuracies - so the dashboard marker matches the patience metric.
        self._post(self._url(pc), {
            "type": "run_end",
            "epoch_last_improved": epoch_last_improved,
            "global_best": {
                "epoch_index": epoch_last_improved,
                "score": global_best_score,
            },
        }, pc)

    def log(self, pc, level, message):
        if level in ("warning", "error") or not pc.get_silent():
            print(message)
        if not self._enabled(pc):
            return
        self._post(self._url(pc), {
            "type": "log",
            "level": level,
            "message": message,
        }, pc)


emitter = DashboardEventEmitter()
