#!/usr/bin/env bash

GEOMETRY=bem_model/sphere_head.geom
CONDUCTIVITIES=bem_model/sphere_head.cond
DIPOLES=bem_model/dipole_locations.txt
EIT_DIPOLES=bem_model/eit_dipole_locations.txt
ELECTRODES=bem_model/sensor_locations.txt
MEG_ELECTRODES=bem_model/meg_sensor_locations.txt
ECOG_ELECTRODES=bem_model/ecog_sensor_locations.txt

# Leadfieldss
EEG_LEADFIELD=leadfields/eeg_leadfield.mat
# EEG_LEADFIELD_ADJOINT=leadfields/eeg_leadfield_adjoint.mat
# ECOG_LEADFIELD=leadfields/ecog_leadfield.mat
MEG_LEADFIELD=leadfields/meg_leadfield.mat
# MEG_LEADFIELD_ADJOINT=leadfields/meg_leadfield_adjoint.mat
EIT_LEADFIELD=leadfields/eit_leadfield.mat
# IP_LEADFIELD=leadfields/ip_leadfield.mat
# EITIP_LEADFIELD=leadfields/eitip_leadfield.mat

# Name temporary matrices
HM=tmp/tmp.hm           # For EEG and MEG
HMINV=tmp/tmp.hm_inv    # For EEG and MEG
DSM=tmp/tmp.dsm         # For EEG and MEG
EIT_DSM=tmp/tmp.eit_dsm # For EIT (cortical sources)
H2EM=tmp/tmp.h2em       # For EEG
# H2ECOGM=tmp/tmp.h2ecogm # For ECoG
H2MM=tmp/tmp.h2mm       # For MEG
DS2MEG=tmp/tmp.ds2mm    # For MEG
EITSM=tmp/tmp.eitsm     # For EIT
# IPHM=tmp/tmp.iphm       # For IP
# IPSM=tmp/tmp.ipsm       # For IP

mkdir -p tmp
mkdir -p leadfields

# Compute EEG gain matrix
om_assemble -HM ${GEOMETRY} ${CONDUCTIVITIES} ${HM}
om_minverser ${HM} ${HMINV}
om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${DSM}
om_assemble -H2EM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${H2EM}
om_gain -EEG ${HMINV} ${DSM} ${H2EM} ${EEG_LEADFIELD}
# or with the adjoint
# om_gain -EEGadjoint ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${HM} ${H2EM} ${EEG_LEADFIELD_ADJOINT}

# # Compute ECoG gain matrix
# om_assemble -H2ECOGM ${GEOMETRY} ${CONDUCTIVITIES} ${ECOG_ELECTRODES} "Brain" ${H2ECOGM}
# om_gain -EEG ${HMINV} ${DSM} ${H2ECOGM} ${ECOG_LEADFIELD}

# Compute MEG gain matrix
om_assemble -H2MM ${GEOMETRY} ${CONDUCTIVITIES} ${MEG_ELECTRODES} ${H2MM}
om_assemble -DS2MM ${DIPOLES} ${MEG_ELECTRODES} ${DS2MEG}
om_gain -MEG ${HMINV} ${DSM} ${H2MM} ${DS2MEG} ${MEG_LEADFIELD}
# # or with the adjoint
# om_gain -MEGadjoint ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${HM} ${H2MM} ${DS2MEG} ${MEG_LEADFIELD_ADJOINT}

# # Compute EIT gain matrix (using cortical dipole sources)
om_assemble -DSM ${GEOMETRY} ${CONDUCTIVITIES} ${EIT_DIPOLES} ${EIT_DSM}
om_assemble -EITSM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${EITSM}
om_gain -EEG ${HMINV} ${EIT_DSM} ${H2EM} ${EIT_LEADFIELD}

# # Compute Internal Potential ....
# om_assemble -H2IPM ${GEOMETRY} ${CONDUCTIVITIES} ${ELECTRODES} ${IPHM}
# # ...for internal dipoles
# om_assemble -DS2IPM ${GEOMETRY} ${CONDUCTIVITIES} ${DIPOLES} ${ELECTRODES} ${IPSM}
# om_gain -IP ${HMINV} ${DSM} ${IPHM} ${IPSM} ${IP_LEADFIELD}
# # ...for boundary-injected current
# om_gain -EITIP ${HMINV} ${EITSM} ${IPHM} ${EITIP_LEADFIELD}