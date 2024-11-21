# %%
# Import necessary libraries
import os
import xarray as xr
from netCDF4 import Dataset
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OPENAI_API_KEY  # Assuming your OpenAI API key is stored in config.py

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# %%
# Define metadata model
class NetCDFMetadata(BaseModel):
    dimensions: dict = Field(..., description="Dimensions of the NetCDF file.")
    variables: List[str] = Field(..., description="Variables available in the NetCDF file.")
    attributes: dict = Field(..., description="Global attributes of the NetCDF file.")
    file_name: str = Field(..., description="The name of the NetCDF file.")

# Function to extract metadata from a netCDF file
def extract_netcdf_metadata(file_path: str) -> NetCDFMetadata:
    with Dataset(file_path, 'r') as nc:
        dimensions = {dim: len(nc.dimensions[dim]) for dim in nc.dimensions}
        variables = list(nc.variables.keys())
        attributes = {attr: nc.getncattr(attr) for attr in nc.ncattrs()}
        file_name = os.path.basename(file_path)
        
        return NetCDFMetadata(
            dimensions=dimensions,
            variables=variables,
            attributes=attributes,
            file_name=file_name
        )

# Function to generate a specific prompt for the LLM based on metadata and user query
def generate_llm_prompt(metadata: NetCDFMetadata, custom_field_request: str) -> str:
    variable_descriptions = []
    for var in metadata.variables:
        if var in metadata.dimensions:
            var_dims = metadata.dimensions[var]
            variable_descriptions.append(f"'{var}' with dimensions {var_dims}")
    
    variable_info = ", ".join(variable_descriptions)
    
    # Generate a dynamic and specific prompt
    prompt = (
        f"You are an expert Python software engineer. You have an xarray dataset loaded with the object named 'data'. "
        f"The dataset contains the following variables: {variable_info}. "
        f"The dimensions and units of the variables are: {metadata.dimensions}. "
        f"Please generate Python code to calculate the {custom_field_request} from the available variables. "
        f"Store the result in an object called 'output'. "
        f"Please use only xarray and numpy libraries. "
        f"Provide ONLY the Python code, without any import lines or comments."
    )
    
    return prompt

# %%
# Function to call OpenAI API and generate Python code
def generate_python_code(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Adjust based on expected response length
            temperature=0.0  # Set to 0 for deterministic output
        )
        # Extract the generated code from the response
        code = response.choices[0].message.content
        return code.strip()  # Remove any leading/trailing whitespace
    except Exception as e:
        return str(e)

# %%
# Example netCDF file path (replace with your file path)
file_path = "/workspaces/llama-extract-metadata-netCDF/gom_t007.nc"  # Replace with your actual file path

# Extract metadata from the netCDF file
metadata = extract_netcdf_metadata(file_path)

# Display extracted metadata
print(metadata)

# %%
# Custom field request for vorticity calculation (from variables U and V)
custom_field_request = "vorticity field from U and V"

# Generate the prompt based on the metadata and user query
prompt = generate_llm_prompt(metadata, custom_field_request)

# Generate Python code using OpenAI API
generated_code = generate_python_code(prompt)

# Output the generated Python code
print("Generated Python Code:\n")
print(generated_code)

# %%
# Assume the generated code provides an 'xarray' dataset called 'data'
# We will manually extract variables `u` and `v` and compute vorticity

# Load the dataset using xarray (replace with your actual file path)
data = xr.open_dataset(file_path)

# Assume the generated code uses variables `u` and `v`
u = data['u']
v = data['v']

# Compute grid spacing for vorticity calculation
dx = u['lon'].diff('lon').mean().values
dy = v['lat'].diff('lat').mean().values

# Calculate vorticity field
vorticity = (v.diff('lon')/dx - u.diff('lat')/dy)

# Store the result in 'output'
output = vorticity.rename('vorticity')

# Display the result
print("Vorticity Field (output):\n")
print(output)

# %%
# Step 1: Import Necessary Libraries
import os
import xarray as xr
from netCDF4 import Dataset
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OPENAI_API_KEY  # Assuming your API key is stored in config.py

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Step 2: Define NetCDF Metadata Model
class NetCDFMetadata(BaseModel):
    dimensions: dict = Field(..., description="Dimensions of the NetCDF file.")
    variables: List[str] = Field(..., description="Variables available in the NetCDF file.")
    attributes: dict = Field(..., description="Global attributes of the NetCDF file.")
    file_name: str = Field(..., description="The name of the NetCDF file.")

# Step 3: Define Function to Extract Metadata from NetCDF
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

# Step 4: Function to Generate LLM Prompt for Python Code
def generate_llm_prompt(metadata: NetCDFMetadata, custom_field_request: str) -> str:
    variable_descriptions = []
    for var in metadata.variables:
        if var in metadata.dimensions:
            var_dims = metadata.dimensions[var]
            variable_descriptions.append(f"'{var}' with dimensions {var_dims}")
    
    variable_info = ", ".join(variable_descriptions)
    
    prompt = (
        f"You are an expert Python software engineer. You have an xarray dataset loaded with the object named 'data'. "
        f"The dataset contains the following variables: {variable_info}. "
        f"The dimensions and units of the variables are: {metadata.dimensions}. "
        f"Please generate the {custom_field_request}. "
        f"The output must be stored in an object called 'output'. "
        f"Please restrict the solution to use only xarray and numpy libraries. "
        f"Please provide ONLY the Python code as an answer, with no import lines or comments. "
        f"Simplify the code as much as possible."
    )
    
    return prompt

# Step 5: Function to Call OpenAI API to Generate Python Code
def generate_python_code(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0  # Deterministic output
        )
        code = response.choices[0].message.content
        return code.strip()
    except Exception as e:
        return str(e)

# Step 6: Load Data and Handle Time Decoding Issue
file_path = "/workspaces/llama-extract-metadata-netCDF/gom_t007.nc"  # Replace with actual path

# Open the dataset with decode_times=False to handle the time units issue
data = xr.open_dataset(file_path, decode_times=False)

# Inspect the dataset to confirm it loaded correctly
print(data)

# Step 7: Extract Metadata
metadata = extract_netcdf_metadata(file_path)

# Step 8: User Query and Generate LLM Prompt
custom_field_request = "vorticity field"  # Example query (you can change it based on the user's needs)
llm_prompt = generate_llm_prompt(metadata, custom_field_request)

# Display the generated prompt
print("Generated LLM Prompt:\n")
print(llm_prompt)

# Step 9: Generate Python Code Using OpenAI API
generated_code = generate_python_code(llm_prompt)

# Display the generated Python code
print("Generated Python Code:\n")
print(generated_code)

# Step 10: Execute the Generated Code
# Assuming the generated code is for calculating vorticity, let's execute it using the loaded dataset

# Manually adapt the generated code to work with the 'data' variable in the notebook
# Example of expected code execution for vorticity (you can modify this if the generated code is different)

# Step 10: Execute the Generated Code
# Assuming the generated code is for calculating vorticity, let's execute it using the loaded dataset

# Extracting the necessary variables
u = data['water_u']  # Corrected variable name
v = data['water_v']  # Corrected variable name

# Calculate grid spacing
dx = u['lon'].diff('lon').mean().values
dy = u['lat'].diff('lat').mean().values

# Calculate vorticity
vorticity = (v.diff('lon') / dx - u.diff('lat') / dy)

# Mask NaN values (optional but can improve result)
vorticity_cleaned = vorticity.where(~vorticity.isnull(), drop=True)

# Store the cleaned vorticity result in the 'output' object
output = vorticity_cleaned.rename('vorticity')

# Print the final result and its metadata
print("Vorticity Calculation Complete")
print(output)



# %%
# Step 11: Use OpenAI to summarize the output (vorticity) result for scientific researchers
def summarize_for_research(output_data: str, model: str = "gpt-4") -> str:
    prompt = f"Summarize the following data with a scientific focus, highlighting the presence of missing values ('nan'), the structure of the dataset, and any significant trends or numerical data points that may appear towards the end of the array:\n\n{output_data}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust to ensure a more detailed response
            temperature=0.7  # Adjust for more or less creativity
        )
        
        # Extract summary from the response
        summary = response.choices[0].message.content
        return summary.strip()  # Remove any leading/trailing whitespace
    except Exception as e:
        return str(e)

# Step 12: Generate the output data (for example, a textual representation of the vorticity result)
# You can focus on key statistics like the presence of 'nan' and trends near the end of the data
output_data = str(output)  # Convert output to string (for summarization)

# Summarize the result using OpenAI, with a focus on scientific relevance
summary = summarize_for_research(output_data)

# Display the summary of the output
print("Scientific Summary of the Output:")
print(summary)


# %%
# Example 2: Query to calculate divergence
custom_field_request = "divergence field from U and V"
prompt = generate_llm_prompt(metadata, custom_field_request)
generated_code = generate_python_code(prompt)
print(f"Generated Python Code for {custom_field_request}:\n")
print(generated_code)




# %%
# Import necessary libraries
import os
import xarray as xr
from netCDF4 import Dataset
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OPENAI_API_KEY  

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# %%
# Define metadata model
class NetCDFMetadata(BaseModel):
    dimensions: dict = Field(..., description="Dimensions of the NetCDF file.")
    variables: List[str] = Field(..., description="Variables available in the NetCDF file.")
    attributes: dict = Field(..., description="Global attributes of the NetCDF file.")
    file_name: str = Field(..., description="The name of the NetCDF file.")

# Function to extract metadata from a netCDF file
def extract_netcdf_metadata(file_path: str) -> NetCDFMetadata:
    with Dataset(file_path, 'r') as nc:
        dimensions = {dim: len(nc.dimensions[dim]) for dim in nc.dimensions}
        variables = list(nc.variables.keys())
        attributes = {attr: nc.getncattr(attr) for attr in nc.ncattrs()}
        file_name = os.path.basename(file_path)
        
        return NetCDFMetadata(
            dimensions=dimensions,
            variables=variables,
            attributes=attributes,
            file_name=file_name
        )

# Function to generate a specific prompt for the LLM based on metadata and user query
def generate_llm_prompt(metadata: NetCDFMetadata, custom_field_request: str) -> str:
    variable_descriptions = []
    for var in metadata.variables:
        if var in metadata.dimensions:
            var_dims = metadata.dimensions[var]
            variable_descriptions.append(f"'{var}' with dimensions {var_dims}")
    
    variable_info = ", ".join(variable_descriptions)
    
    # Generate a dynamic and specific prompt
    prompt = (
        f"You are an expert Python software engineer. You have an xarray dataset loaded with the object named 'data'. "
        f"The dataset contains the following variables: {variable_info}. "
        f"The dimensions and units of the variables are: {metadata.dimensions}. "
        f"Please generate Python code to calculate the {custom_field_request} from the available variables. "
        f"Store the result in an object called 'output'. "
        f"Please use only xarray and numpy libraries. "
        f"Provide ONLY the Python code, without any import lines or comments."
    )
    
    return prompt

# %%
# Function to call OpenAI API and generate Python code
def generate_python_code(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150, 
            temperature=0.0  
        )

        # Extract the generated code from the response
        code = response.choices[0].message.content
        return code.strip()  
    except Exception as e:
        return str(e)

# %%
# Example netCDF file path (replace with your file path)
file_path = "/workspaces/llama-extract-metadata-netCDF/gom_t007.nc" 

# Extract metadata from the netCDF file
metadata = extract_netcdf_metadata(file_path)

# Display extracted metadata
print(metadata)

# %%
# Custom field request for vorticity calculation (from variables U and V)
custom_field_request = "vorticity field from U and V"

# Generate the prompt based on the metadata and user query
prompt = generate_llm_prompt(metadata, custom_field_request)

# Generate Python code using OpenAI API
generated_code = generate_python_code(prompt)

# Output the generated Python code
print("Generated Python Code:\n")
print(generated_code)

# %%
data = xr.open_dataset(file_path)

# Assume the generated code uses variables `u` and `v`
u = data['u']
v = data['v']

# Compute grid spacing for vorticity calculation
dx = u['lon'].diff('lon').mean().values
dy = v['lat'].diff('lat').mean().values

# Calculate vorticity field
vorticity = (v.diff('lon')/dx - u.diff('lat')/dy)

# Store the result in 'output'
output = vorticity.rename('vorticity')

# Display the result
print("Vorticity Field (output):\n")
print(output)

# %%
# Step 1: Import Necessary Libraries
import os
import xarray as xr
from netCDF4 import Dataset
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from config import OPENAI_API_KEY 

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Step 2: Define NetCDF Metadata Model
class NetCDFMetadata(BaseModel):
    dimensions: dict = Field(..., description="Dimensions of the NetCDF file.")
    variables: List[str] = Field(..., description="Variables available in the NetCDF file.")
    attributes: dict = Field(..., description="Global attributes of the NetCDF file.")
    file_name: str = Field(..., description="The name of the NetCDF file.")

# Step 3: Define Function to Extract Metadata from NetCDF
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

# Step 4: Function to Generate LLM Prompt for Python Code
def generate_llm_prompt(metadata: NetCDFMetadata, custom_field_request: str) -> str:
    variable_descriptions = []
    for var in metadata.variables:
        if var in metadata.dimensions:
            var_dims = metadata.dimensions[var]
            variable_descriptions.append(f"'{var}' with dimensions {var_dims}")
    
    variable_info = ", ".join(variable_descriptions)
    
    prompt = (
        f"You are an expert Python software engineer. You have an xarray dataset loaded with the object named 'data'. "
        f"The dataset contains the following variables: {variable_info}. "
        f"The dimensions and units of the variables are: {metadata.dimensions}. "
        f"Please generate the {custom_field_request}. "
        f"The output must be stored in an object called 'output'. "
        f"Please restrict the solution to use only xarray and numpy libraries. "
        f"Please provide ONLY the Python code as an answer, with no import lines or comments. "
        f"Simplify the code as much as possible."
    )
    
    return prompt

# Step 5: Function to Call OpenAI API to Generate Python Code
def generate_python_code(prompt: str, model: str = "gpt-4") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.0  
        )
        code = response.choices[0].message.content
        return code.strip()
    except Exception as e:
        return str(e)

# Step 6: Load Data and Handle Time Decoding Issue
file_path = "/workspaces/llama-extract-metadata-netCDF/gom_t007.nc"  

# Open the dataset with decode_times=False to handle the time units issue
data = xr.open_dataset(file_path, decode_times=False)

# Inspect the dataset to confirm it loaded correctly
print(data)

# Step 7: Extract Metadata
metadata = extract_netcdf_metadata(file_path)

# Step 8: User Query and Generate LLM Prompt
custom_field_request = "vorticity field"  # Example query (you can change it based on the user's needs)
llm_prompt = generate_llm_prompt(metadata, custom_field_request)

# Display the generated prompt
print("Generated LLM Prompt:\n")
print(llm_prompt)

# Step 9: Generate Python Code Using OpenAI API
generated_code = generate_python_code(llm_prompt)

# Display the generated Python code
print("Generated Python Code:\n")
print(generated_code)


# Extracting the necessary variables
u = data['water_u']  # Corrected variable name
v = data['water_v']  # Corrected variable name

# Calculate grid spacing
dx = u['lon'].diff('lon').mean().values
dy = u['lat'].diff('lat').mean().values

# Calculate vorticity
vorticity = (v.diff('lon') / dx - u.diff('lat') / dy)

# Mask NaN values (optional but can improve result)
vorticity_cleaned = vorticity.where(~vorticity.isnull(), drop=True)

# Store the cleaned vorticity result in the 'output' object
output = vorticity_cleaned.rename('vorticity')

# Print the final result and its metadata
print("Vorticity Calculation Complete")
print(output)



# %%
# Step 11: Use OpenAI to summarize the output (vorticity) result for scientific researchers
def summarize_for_research(output_data: str, model: str = "gpt-4") -> str:
    prompt = f"Summarize the following data with a scientific focus, highlighting the presence of missing values ('nan'), the structure of the dataset, and any significant trends or numerical data points that may appear towards the end of the array:\n\n{output_data}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  
            temperature=0.7  
        )
        
        # Extract summary from the response
        summary = response.choices[0].message.content
        return summary.strip()  
    except Exception as e:
        return str(e)

# Step 12: Generate the output data (for example, a textual representation of the vorticity result)
output_data = str(output)  

# Summarize the result using OpenAI, with a focus on scientific relevance
summary = summarize_for_research(output_data)

# Display the summary of the output
print("Scientific Summary of the Output:")
print(summary)


# %%
# Example 2: Query to calculate divergence
custom_field_request = "divergence field from U and V"
prompt = generate_llm_prompt(metadata, custom_field_request)
generated_code = generate_python_code(prompt)
print(f"Generated Python Code for {custom_field_request}:\n")
print(generated_code)
