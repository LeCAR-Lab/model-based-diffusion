import numpy as np
import xml.etree.ElementTree as ET

# XML content string (simplified for this example)
# load walk_ref.xml file
with open('walk_ref.xml', 'r') as file:
    xml_content = file.read()

# Parse the XML using ElementTree
root = ET.fromstring(xml_content)

# Initialize an empty list to collect all mpos data
all_mpos = []

# Iterate through each key element in the XML
for key_element in root.findall('.//key'):
    # Extract the mpos attribute and convert it to a float list
    mpos_list = list(map(float, key_element.attrib['mpos'].split()))
    mpos = np.array(mpos_list).reshape(-1, 3)
    # Append to the overall mpos list
    all_mpos.append(mpos)

# Convert the list of mpos data to a numpy array
mpos_array = np.stack(all_mpos)

# Save the numpy array to a .npy file
np.save('walk_ref.npy', mpos_array)