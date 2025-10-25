from dataclasses import dataclass, asdict
from typing import Optional, Dict
import json
import hashlib


@dataclass
class Parameters:
    """
    Parameters structure for SVD analysis.

    Attributes
    ----------
    num_sensors : int, optional
        Number of sensors used in the measurement
    grid_resolution : float, optional
        Resolution of the computational grid
    num_brain_grid_points : int, optional
        Number of grid points in the brain model
    time_resolution : float, optional
        Temporal resolution of the measurement
    comment : str, optional
        Additional comment or description
    """

    num_sensors: Optional[int] = None
    grid_resolution_mm: Optional[float] = None
    num_brain_grid_points: Optional[int] = None
    time_resolution: Optional[float] = None
    comment: Optional[str] = None
    noise_full_brain: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "Parameters":
        """Create Parameters object from dictionary."""
        return cls(
            num_sensors=data.get("num_sensors"),
            grid_resolution_mm=data.get("grid_resolution_mm")
            or data.get("grid_resolution"),
            num_brain_grid_points=data.get("num_brain_grid_points"),
            time_resolution=data.get("time_resolution"),
            comment=data.get("comment"),
            noise_full_brain=data.get("noise_full_brain"),
        )

    def __str__(self) -> str:
        fields = []
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if value is not None:
                fields.append(f"{field}={value!r}")
        return f"Parameters({', '.join(fields)})" if fields else "Parameters()"
    
    def get_hash(self) -> str:
        params_dict = asdict(self)
        # Sort keys for consistent hashing
        params_str = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict:
        """Return dictionary of only non-None fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}
