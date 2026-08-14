"""Quantum machine-learning models."""

from .qcnn import qcnn_expectation, train_qcnn_spsa

__all__ = ["qcnn_expectation", "train_qcnn_spsa"]
