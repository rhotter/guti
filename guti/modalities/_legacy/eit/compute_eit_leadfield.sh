#!/usr/bin/env bash

# EIT-specific leadfield computation
GEOMETRY=bem_model/eit/sphere_head.geom
CONDUCTIVITIES=bem_model/eit/sphere_head.cond
EIT_DIPOLES=bem_model/eit/eit_dipole_locations.txt
ELECTRODES=bem_model/eit/sensor_locations.txt

# Output
EIT_LEADFIELD=leadfields/eit/eit_leadfield.mat

# Temporary matrices
HM=tmp/eit/tmp.hm
HMINV=tmp/eit/tmp.hm_inv
H2EM=tmp/eit/tmp.h2em
EITSM=tmp/eit/tmp.eitsm

mkdir -p tmp/eit
mkdir -p leadfields/eit

# Compute EIT gain matrix (using cortical dipole sources)
om_assemble -HM ${GEOMETRY} ${CONDUCTIVITIES} ${HM}
om_minverser ${HM} ${HMINV}
om_assemble -H2EM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${H2EM}
# om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${EIT_DIPOLES} ${EIT_DSM}
om_assemble -EITSM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${EITSM}
om_gain -EEG ${HMINV} ${EITSM} ${H2EM} ${EIT_LEADFIELD}
