# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2018-10-22
### Fixed
- `identify.get_apoz` fails to identify all generators and resulted
  high memory usage (possibly an infinite loop) when calculating layer
  outputs
- `identify.get_apoz` calculates incorrect apoz when layers are re-used

## [0.1.2] - 2018-10-04
### Fixed
- Bug when pruning Conv layers with smaller input than filter shape
  (identified in a VGG-like architecture).

## [0.1.1] - 2018-08-07
### Fixed
- Updated to work with keras >= 2.2.0
- No longer triggers numpy FutureWarning: "Using a non-tuple sequence
  for multidimensional indexing is deprecated".
- Updated flowers example to work with latest keras and tensorflow. It
  probably won't work with old versions now.

## [0.1.0] - 2018-05-15
### Added
- Deleting all neurons in a layer now removes the whole branch.
  If there is only one branch, this will cause an error to be raised.
- Enabled resuming pruning from the last checkpoint in `inception_flowers_prune`
  example. This is now the default behaviour.
- Added a changelog!

### Changed
- Updated to work with Keras >= 2.1.3.
- Massively reduced test time.
- Updated tox.ini to test many combinations of keras and tensorflow versions.

### Deprecated
- Support for Keras < 2.1.3 will be removed in a future release.

### Fixed
- Shared layers are no longer broken when `delete_channels` is applied to
  upstream layers·
- Added memory cleanup to `inception_flowers_prune` example; prevents memory leak.




[Unreleased]: https://github.com/BenWhetton/keras-surgeon/compare/0.1.0...HEAD
[0.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/0.0.1...0.1.0
