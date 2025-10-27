#!/usr/bin/env bash

# EEG-specific leadfield computation
GEOMETRY=guti/modalities/bem_model/eeg/sphere_head.geom
CONDUCTIVITIES=guti/modalities/bem_model/eeg/sphere_head.cond
DIPOLES=guti/modalities/bem_model/eeg/dipole_locations.txt
ELECTRODES=guti/modalities/bem_model/eeg/sensor_locations.txt

# Output
EEG_LEADFIELD=guti/modalities/leadfields/eeg/eeg_leadfield.mat

# Temporary matrices
HM=guti/modalities/tmp/eeg/tmp.hm
HMINV=guti/modalities/tmp/eeg/tmp.hm_inv
DSM=guti/modalities/tmp/eeg/tmp.dsm
H2EM=guti/modalities/tmp/eeg/tmp.h2em

mkdir -p guti/modalities/tmp/eeg
mkdir -p guti/modalities/leadfields/eeg

# Compute EEG gain matrix
om_assemble -HM ${GEOMETRY} ${CONDUCTIVITIES} ${HM}
om_minverser ${HM} ${HMINV}
om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${DSM}
om_assemble -H2EM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${H2EM}
om_gain -EEG ${HMINV} ${DSM} ${H2EM} ${EEG_LEADFIELD}
