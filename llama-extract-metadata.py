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
extract_netcdf_metadata('/workspaces/llama-extract-metadata-netCDF/gom_t007.nc')

# %%
# Corrected function to avoid repeated "field" in the prompt
def generate_llm_prompt(metadata: NetCDFMetadata, custom_field_request: str) -> str:
    # Extract variable names and dimensions
    variable_descriptions = []
    for var in metadata.variables:
        if var in metadata.dimensions:
            var_dims = metadata.dimensions[var]
            variable_descriptions.append(f"'{var}' with dimensions {var_dims}")
    
    # Create a string description of the dataset fields and their dimensions
    variable_info = ", ".join(variable_descriptions)
    
    # Create the dynamic prompt
    prompt = (
        f"You are an expert Python software engineer. You have an xarray dataset loaded with the object named 'data'. "
        f"The dataset contains the following variables: {variable_info}. "
        f"The dimensions and units of the variables are: {metadata.dimensions}. "
        f"Please generate the {custom_field_request}. "  # Adjusted this part to avoid repetition
        f"The output must be stored in an object called 'output'. "
        f"Please restrict the solution to use only xarray and numpy libraries. "
        f"Please provide ONLY the Python code as an answer, with no import lines or comments. "
        f"Simplify the code as much as possible."
    )
    
    return prompt

# %% 
# Example of how to use this in the notebook
file_path = "/workspaces/llama-extract-metadata-netCDF/gom_t007.nc"  # Provide your actual file path here
metadata = extract_netcdf_metadata(file_path)

# User's custom field request, e.g., "vorticity field from U and V"
custom_field_request = "vorticity field"

# Generate the LLM prompt based on the extracted metadata and user request
llm_prompt = generate_llm_prompt(metadata, custom_field_request)

# Display the generated prompt
print("Generated LLM Prompt:\n")
print(llm_prompt)

# %%
# %% [markdown]
# ## Step 1: Import Libraries and Set API Key
# %%
import os
from openai import OpenAI  # New import style
from config import OPENAI_API_KEY  # Assuming your API key is stored in config.py

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# %% [markdown]
# ## Step 2: Define the Function to Call the OpenAI API
# %%
def generate_python_code(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,  # Adjust based on expected response length
            temperature=0.0  # Set to 0 for deterministic output
        )
        # Extract the generated code from the response
        code = response.choices[0].message.content
        return code.strip()  # Remove any leading/trailing whitespace
    except Exception as e:
        return str(e)

# %% [markdown]
# ## Step 3: Generate the Prompt and Call the API
# %%
# Assuming you have the metadata already extracted
metadata = extract_netcdf_metadata("/workspaces/llama-extract-metadata-netCDF/gom_t007.nc")  # Replace with actual path
custom_field_request = "vorticity field"  # Custom field request

# Generate the prompt
prompt = generate_llm_prompt(metadata, custom_field_request)

# Call the OpenAI API to generate the Python code
generated_code = generate_python_code(prompt)

# Output the generated code
print("Generated Python Code:\n")
print(generated_code)


# %%



