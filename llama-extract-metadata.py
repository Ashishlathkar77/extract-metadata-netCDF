# %%
import os
import xarray as xr
from netCDF4 import Dataset
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# %%
class NetCDFMetadata(BaseModel):
    dimensions: dict = Field(..., description="Dimensions of the NetCDF file.")
    variables: List[str] = Field(..., description="Variables available in the NetCDF file.")
    attributes: dict = Field(..., description="Global attributes of the NetCDF file.")
    file_name: str = Field(..., description="The name of the NetCDF file.")

# %%
def extract_netcdf_metadata(file_path: str) -> NetCDFMetadata:
    with Dataset(file_path, 'r') as nc:
        dimensions = {dim: len(nc.dimensions[dim]) for dim in nc.dimensions}
        variables = list(nc.variables.keys())
        attributes = {attr: nc.getncattr(attr) for attr in nc.ncattrs()}
        file_name = os.path.basename(file_path)
        
        return NetCDFMetadata(
            dimensions = dimensions,
            variables = variables,
            attributes = attributes,
            file_name = file_name
        )

# %%



