from __future__ import annotations
from typing import Self
from dataclasses import dataclass

@dataclass
class Beat:
	location: float
	is_not_synthetic: bool
	time_per_beat: float
	prominence_avg: float
	prominence: float

@dataclass
class GreedyBeat(Beat):
	tracker_confidence: float
	tracker_time_per_beat: float

	@classmethod
	def from_Beat(cls, beat: Beat, tracker_confidence: float, tracker_time_per_beat: float) -> Self:
		return cls(beat.location, beat.is_not_synthetic, beat.time_per_beat, beat.prominence_avg, beat.prominence, tracker_confidence, tracker_time_per_beat)

