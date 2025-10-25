# Domain Description 1.1

Interfaces 3

Interface Brain: "brain_sphere.tri"
Interface Skull: "skull_sphere.tri"
Interface Scalp: "scalp_sphere.tri"

Domains 4

Domain Brain: -Brain
Domain Skull: -Skull +Brain
Domain Scalp: -Scalp +Skull
Domain Air: +Scalp
