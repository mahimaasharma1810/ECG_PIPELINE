"""Independent classification axes.

Three axes, each with its own refusal, rather than one flat label. A flat
label forces a choice between things that co-occur and cannot express partial
confidence - which matters here, because on device data the RATE is currently
trustworthy while REGULARITY is not.

    rate.py    bradycardic / normal / tachycardic / undetermined
    events.py  pauses and ectopy BURDEN (never type)
    combine.py maps the axes to a severity band, incl. the regular-tachycardia
               rule that a regularity-only view misses
"""
from .rate import RateAxis, assess_rate, rate_trustworthy
from .events import EventsAxis, assess_events, count_ectopic_couplets
from .combine import combine_axes, CombinedAssessment

__all__ = ["RateAxis", "assess_rate", "rate_trustworthy", "EventsAxis", "assess_events",
           "count_ectopic_couplets", "combine_axes", "CombinedAssessment"]
