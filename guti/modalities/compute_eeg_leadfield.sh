#!/usr/bin/env bash

# EEG-specific leadfield computation
GEOMETRY=bem_model/eeg/sphere_head.geom
CONDUCTIVITIES=bem_model/eeg/sphere_head.cond
DIPOLES=bem_model/eeg/dipole_locations.txt
ELECTRODES=bem_model/eeg/sensor_locations.txt

# Output
EEG_LEADFIELD=leadfields/eeg/eeg_leadfield.mat

# Temporary matrices
HM=tmp/eeg/tmp.hm
HMINV=tmp/eeg/tmp.hm_inv
DSM=tmp/eeg/tmp.dsm
H2EM=tmp/eeg/tmp.h2em

mkdir -p tmp/eeg
mkdir -p leadfields/eeg

# Compute EEG gain matrix
om_assemble -HM ${GEOMETRY} ${CONDUCTIVITIES} ${HM}
om_minverser ${HM} ${HMINV}
om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${DSM}
om_assemble -H2EM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${H2EM}
om_gain -EEG ${HMINV} ${DSM} ${H2EM} ${EEG_LEADFIELD}
