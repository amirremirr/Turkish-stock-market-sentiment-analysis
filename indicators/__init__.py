"""Descriptive financial-news indicators computed from scored headlines.

Every module here is storage-independent: it takes records and returns rows, so
the arithmetic can be tested without a database. All time-series normalization
uses strictly prior observations.
"""
