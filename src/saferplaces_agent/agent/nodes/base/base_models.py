from typing import Optional, Union, List, Dict, Any, Literal
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator



class BBox(BaseModel):
    """
    Bounding box in EPSG:4326 (WGS84).
    - `west` = min longitude
    - `south` = min latitude
    - `east` = max longitude
    - `north` = max latitude
    """
    west: float = Field(..., description="Minimum longitude (degrees), e.g., 10.0")
    south: float = Field(..., description="Minimum latitude (degrees), e.g., 44.0")
    east: float = Field(..., description="Maximum longitude (degrees), e.g., 12.0")
    north: float = Field(..., description="Maximum latitude (degrees), e.g., 46.0")

    def __str__(self):
        return f"{{\"west\": {self.west}, \"south\": {self.south}, \"east\": {self.east}, \"north\": {self.north}}}"
    
    def to_list(self) -> List[float]:
        """
        Convert the bounding box to a list [west, south, east, north].
        """
        return [self.west, self.south, self.east, self.north]
    
    def lat_range(self) -> List[float]:
        """
        Get the latitude range as [south, north].
        """
        return [self.south, self.north]
    
    def long_range(self) -> List[float]:
        """
        Get the longitude range as [west, east].
        """
        return [self.west, self.east]