# Changelog

## [Unreleased]

- Fused concrete-sequence `dollar_bars` construction into one validated streaming state machine while preserving every `Bar` field and error.
- Added a deterministic end-to-end dollar-bar and causal-normalisation benchmark with exact output checksums.


## [1.0.0] - 2026-08-12

First stable release.

- Activity-based market data bars and strictly causal normalisation transforms.
- Added a deterministic property-based suite for bar invariants, conservation, and causal transforms.
- Documented mutation testing: 1,123 of 1,208 mutants killed (92.96%), with the remaining 85 reviewed as behavior-equivalent.
- Adopted strict mypy checking for the typed public package.
- Expanded CI across Linux and macOS on Python 3.11–3.13.
