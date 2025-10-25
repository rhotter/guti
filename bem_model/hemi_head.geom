# Domain Description 1.1

Interfaces 3

Interface Brain: "brain_hemi.tri"
Interface Skull: "skull_hemi.tri"
Interface Scalp: "scalp_hemi.tri"

Domains 4

Domain Brain: -Brain
Domain Skull: -Skull +Brain
Domain Scalp: -Scalp +Skull
Domain Air: +Scalp
