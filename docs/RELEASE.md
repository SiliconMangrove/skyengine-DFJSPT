# v0.3.0 — Initial public release

Release date: 2026-10-21.

Documentation correction (2026-10-21): the web platform's pipelined PPO trainer requires CUDA. CPU training uses the generic headless command-line trainer. The maintained README and usage guide clarify this distinction; the original v0.3.0 tag is unchanged.

This release packages the CTDE-PPO controller from commit
`6d1787c0b0ffa7be78e89134165366f796e5bbae`, paired with SkyEngine commit
`800fc454dc29124e95e04e9abc58db3d87fd0383`.

The algorithm implementation, historical training configurations and existing tests are preserved. Release preparation adds the Apache-2.0 license, bilingual README, self-contained usage documentation and explicit Python packaging metadata. Python support is aligned with the companion environment (3.11), and the NumPy lower bound includes its locked 1.23.5 version.

## Recorded execution evidence

The paired version was exercised on 2026-09-24:

- Nine targeted algorithm checks passed, covering action probabilities, gradient direction, buffer admission, stale operations, route conflicts, PPO diagnostics and cancellation.
- Two 150-step training samples reproduced collected action log probabilities with zero recorded discrepancy and completed finite-valued PPO updates.
- Four complete delivery runs finished in 1098, 472, 1128 and 2538 simulation steps.
- A subsequent training snapshot recorded 884 completed episodes out of 893, and each of three model snapshots completed all 30 validation episodes in the fixed panel.

These are historical execution records for the pinned pair, not a new release-day test run or a guarantee for every instance. One historical latency sample exceeded the 300 ms cooperative budget on 4 of 150 steps, with a maximum of 313.9 ms. Task completion and algorithm quality should be measured on the user's chosen workload.

## Included components

- Graph policy and recurrent state, executable transaction candidate search, transport planning and reward model.
- PPO learner, graph batching, existing regression tests and profiling utility.
- Headless training configurations and instructions for the companion platform's training, model evaluation and replay.

Synthetic datasets and the formal factory environment are supplied by the pinned public SkyEngine repository. Model weights are produced by training; no pretrained weights are distributed in this release.
