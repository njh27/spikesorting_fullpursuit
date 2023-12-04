# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [1.1.0] - 2023-12-04

### Added
- Option to base the core cluster comparison test on a matched sample of nearest neighboring datapoints so that large clusters do not iteratively consume small ones
- Option to revert cluster merge split decisions that decrease the distance between clusters, which typically indicates a terrible error

### Changed
- merge_test in sort.py has flag to perform comparisons between clusters using matched sample sizes of cluster points
- merge_test in sort.py has flag to check the nearest neighbor distances using BallTree package in sklearn.neighbors

