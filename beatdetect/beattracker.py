from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math

from .data import Beat, GreedyBeat

MIN_REL_PROMINENCE = 0.25

class BeatTracker:
	beats: list[Beat]
	greedy_beats: list[GreedyBeat]
	timestep: float
	sigma: float
	lam: float
	time_per_beat: float
	last_prom: float
	confidence: float
	used: bool
	greedy_continuity: float # FIXME this does not really belong here

	# both beat_loc and time_per_beat are given in timesteps, i.e. not neccessarily msec
	def __init__(self, timestep: float, sigma: float, lam: float, beat_loc: float, time_per_beat: float, confidence: float, last_prom: float, beats: list[Beat]):
		self.timestep = timestep
		self.sigma = sigma
		self.lam = lam
		self.beat_loc = beat_loc
		self.time_per_beat = time_per_beat
		self.last_prom = last_prom
		self.confidence = confidence
		self.beats = beats[-10:] # limit beat array length
		self.used = False

	# FIXME is this even used outside of greedy tracking?
	def tpb(self) -> float:
		if len(self.beats) > 2:
			first = max(0, len(self.beats) - 1 - 8)
			last = len(self.beats)-1
			return (self.beats[last].location - self.beats[first].location) / (last-first)
		else:
			return self.time_per_beat

	# returns interval where beats are expected as (begin, end) tuple; unit = timesteps
	def search_interval(self) -> tuple[float, float]:
		expected_loc = self.beat_loc + self.time_per_beat
		win_timesteps = self.sigma * 4
		return (expected_loc - win_timesteps, expected_loc+win_timesteps)

	def next_beat(self, peak_locs: Sequence[float], peak_prominences: Sequence[float]) -> list[BeatTracker]:
		if self.time_per_beat < 50/1000/self.timestep:
			# kill degenerate beat detectors that suggest an infinite tempo
			return []

		expected_loc = self.beat_loc + self.time_per_beat

		@dataclass
		class Peak:
			relevance: float  # normalized posterior probability
			location: float   # beat location in timesteps
			is_not_synthetic: bool
			prominence: float # peak prominence

		peaks: list[Peak] = []
		for loc, prom in zip(peak_locs, peak_prominences):
			t_diff = loc - expected_loc
			relevance = math.exp(-0.5*(t_diff/self.sigma)**2) / (self.lam * math.exp(-self.lam * prom))
			if loc > self.beat_loc + self.time_per_beat*0.1:
				peaks.append(Peak(
					relevance=relevance,
					location=loc,
					is_not_synthetic=True,
					prominence=prom))

		peaks.append(Peak(
			relevance=1 / (self.lam * math.exp(-self.lam * self.last_prom * MIN_REL_PROMINENCE)),
			location=expected_loc,
			is_not_synthetic=False,
			prominence=0))

		# normalize peak relevances (to sum=1)
		relevance_sum = sum([p.relevance for p in peaks])
		for p in peaks: p.relevance /= relevance_sum

		peaks = sorted(peaks, key=lambda p: p.relevance)[::-1]

		alpha = 0
		confidence = self.confidence * (1-alpha) + 1 * alpha

		tpb_alpha = 0.7 # 0.98
		prom_alpha = 0.5
		result: list[BeatTracker] = []
		for p in peaks:
			tpb_new = tpb_alpha * self.time_per_beat + (1-tpb_alpha)*(p.location - self.beats[-1].location)
			conf_new = confidence * p.relevance
			prom_new = prom_alpha*self.last_prom + (1-prom_alpha)*p.prominence
			result.append(BeatTracker(self.timestep, self.sigma, self.lam, p.location, tpb_new, conf_new, prom_new, self.beats + [Beat(p.location, p.is_not_synthetic, tpb_new, prom_new, p.prominence)]))
		return result


