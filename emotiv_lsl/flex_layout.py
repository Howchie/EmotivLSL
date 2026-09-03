"""Location labels used by the full EPOC Flex 10-20 calibration image."""

from __future__ import annotations


# This is the row order on images/10-20.png.  Keeping the list in one module
# lets the calibrator and the eventual Flex quality viewer use identical keys.
FLEX_10_20_LOCATIONS = (
    "Fpz",
    "Fp1", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FC8", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO3", "POz", "PO4", "PO8", "PO10",
    "O1", "Oz", "O2",
    "O9", "Iz", "O10",
)

