#!/usr/bin/env bash
# EEG lead-field computation via OpenMEEG.
#
# Usage: compute_eeg_leadfield.sh <model_dir> <out_dir>
#   <model_dir>  directory holding sphere_head.geom/.cond, dipole_locations.txt,
#                sensor_locations.txt (as written by guti.core.create_eeg_bem_model)
#   <out_dir>    directory to write eeg_leadfield.mat (and scratch matrices)
set -e

MODEL_DIR="${1:?usage: compute_eeg_leadfield.sh <model_dir> <out_dir>}"
OUT_DIR="${2:?usage: compute_eeg_leadfield.sh <model_dir> <out_dir>}"

GEOMETRY="$MODEL_DIR/sphere_head.geom"
CONDUCTIVITIES="$MODEL_DIR/sphere_head.cond"
DIPOLES="$MODEL_DIR/dipole_locations.txt"
ELECTRODES="$MODEL_DIR/sensor_locations.txt"

EEG_LEADFIELD="$OUT_DIR/eeg_leadfield.mat"

TMP="$OUT_DIR/tmp"
mkdir -p "$TMP"
HM="$TMP/tmp.hm"
HMINV="$TMP/tmp.hm_inv"
DSM="$TMP/tmp.dsm"
H2EM="$TMP/tmp.h2em"

# Compute EEG gain matrix
om_assemble -HM "$GEOMETRY" "$CONDUCTIVITIES" "$HM"
om_minverser "$HM" "$HMINV"
om_assemble -DSM "$GEOMETRY" "$CONDUCTIVITIES" "$DIPOLES" "$DSM"
om_assemble -H2EM "$GEOMETRY" "$CONDUCTIVITIES" "$ELECTRODES" "$H2EM"
om_gain -EEG "$HMINV" "$DSM" "$H2EM" "$EEG_LEADFIELD"
