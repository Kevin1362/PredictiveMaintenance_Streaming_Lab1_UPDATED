from dataclasses import dataclass

@dataclass
class RuleConfig:
    minC: float   # alert threshold (positive residual)
    maxC: float   # error threshold (positive residual)
    T: float      # seconds required continuously

def detect_events_for_axis(time_s, deviation, axis_name: str, cfg: RuleConfig):
    """
    Detect sustained runs above thresholds.

    Alert: deviation >= minC for >= T seconds continuously (and < maxC)
    Error: deviation >= maxC for >= T seconds continuously
    """
    events = []

    def flush_event(kind, start_idx, end_idx, threshold):
        start_time = float(time_s[start_idx])
        end_time = float(time_s[end_idx])
        duration = float(end_time - start_time)
        max_dev = float(max(deviation[start_idx:end_idx+1]))
        events.append({
            "axis_name": axis_name,
            "event_type": kind,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration,
            "threshold": float(threshold),
            "max_deviation": max_dev
        })

    in_alert = False
    in_error = False
    alert_start = None
    error_start = None

    for i in range(len(time_s)):
        d = deviation[i]

        # ERROR runs
        if d >= cfg.maxC:
            if not in_error:
                in_error = True
                error_start = i
        else:
            if in_error:
                start_i = error_start
                end_i = i - 1
                if time_s[end_i] - time_s[start_i] >= cfg.T:
                    flush_event("ERROR", start_i, end_i, cfg.maxC)
                in_error = False
                error_start = None

        # ALERT runs (only between minC and maxC)
        if cfg.minC <= d < cfg.maxC:
            if not in_alert:
                in_alert = True
                alert_start = i
        else:
            if in_alert:
                start_i = alert_start
                end_i = i - 1
                if time_s[end_i] - time_s[start_i] >= cfg.T:
                    flush_event("ALERT", start_i, end_i, cfg.minC)
                in_alert = False
                alert_start = None

    # finalize tail
    if in_error:
        start_i = error_start
        end_i = len(time_s) - 1
        if time_s[end_i] - time_s[start_i] >= cfg.T:
            flush_event("ERROR", start_i, end_i, cfg.maxC)

    if in_alert:
        start_i = alert_start
        end_i = len(time_s) - 1
        if time_s[end_i] - time_s[start_i] >= cfg.T:
            flush_event("ALERT", start_i, end_i, cfg.minC)

    return events
