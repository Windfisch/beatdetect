# Beatdetect

Audio-in MIDI-clock-out.

Performs beat detection on music with the following rough steps:

- Fourier-transform the signal into frequency components
- Perform onset detection by finding rapid increases in energy in any
  frequency band.
- Lock onto a beat and track it by following onsets.
- Actually do this with multiple candidates at the same time, and select
  best one.

Details of how the algorithm works are in this
[german talk at EH23](https://media.ccc.de/v/eh23-live-beaterkennung-in-musik).

The code is largely undocumented yet. Works for me™.

Some options can be set via CLI, others are hard-coded in variables.
check the `*latency` variables in live.py for example.

## Caveats / good-to-know

- The goal is not to be 100% accurate, but to be good enough to track the tempo
  most of the time, and allow the user to guide the algorithm when tracking fails.
- Tempo fluctuates by ±0.5bpm. Consumers that extrapolate the tempo should use
  appropriate filtering.
- Python is not very real-time-capable. Use large buffer sizes. The additional
  latency should not hurt, as long it's < 1 beat. (We'll be late for the "now"
  beat anyway, so we can take our time until the next beat.)
- Performance seems to degrade over the course of 1-2 hours. Restart the program
  to fix that. (Again, works for me™, YMMV.)
