#!/usr/bin/env bash

# MEG-specific leadfield computation
GEOMETRY=bem_model/meg/sphere_head.geom
CONDUCTIVITIES=bem_model/meg/sphere_head.cond
DIPOLES=bem_model/meg/dipole_locations.txt
MEG_ELECTRODES=bem_model/meg/meg_sensor_locations.txt

# Output
MEG_LEADFIELD=leadfields/meg/meg_leadfield.mat

# Temporary matrices
HM=tmp/meg/tmp.hm
HMINV=tmp/meg/tmp.hm_inv
DSM=tmp/meg/tmp.dsm
H2MM=tmp/meg/tmp.h2mm
DS2MEG=tmp/meg/tmp.ds2mm

mkdir -p tmp/meg
mkdir -p leadfields/meg

# Compute MEG gain matrix
om_assemble -HM ${GEOMETRY} ${CONDUCTIVITIES} ${HM}
om_minverser ${HM} ${HMINV}
om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${DSM}
om_assemble -H2MM ${GEOMETRY} ${CONDUCTIVITIES} ${MEG_ELECTRODES} ${H2MM}
om_assemble -DS2MM ${DIPOLES} ${MEG_ELECTRODES} ${DS2MEG}
om_gain -MEG ${HMINV} ${DSM} ${H2MM} ${DS2MEG} ${MEG_LEADFIELD}
