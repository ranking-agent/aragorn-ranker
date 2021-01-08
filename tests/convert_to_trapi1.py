import os
import json
from reasoner_converter.upgrading import upgrade_Query
from reasoner_converter.downgrading import downgrade_Query

# declare input and output data directories
in_files_path: str = 'InputJson_0.9.2'
out_files_path: str = 'InputJson_1.0'

# get the path of where this file is. everything is relative to that
this_path: str = os.path.dirname(os.path.realpath(__file__))

# declare the input and output directory names
in_dir_name: str = os.path.join(this_path, in_files_path)
out_dir_name: str = os.path.join(this_path, out_files_path)

# get the list of json files in the input directory
files: list = [file for file in os.listdir(in_dir_name) if file.endswith('.json')]

# for each json file found
for file in files:
    # open the input and output files
    with open(os.path.join(in_dir_name, file), 'r') as in_file, open(os.path.join(out_dir_name, file), 'w') as out_file:
        # load the json input data
        data: dict = json.load(in_file)

        # upgrade the whole message
        upgraded_data: dict = upgrade_Query(data)

        # output the upgraded data into the output file
        json.dump(upgraded_data, out_file, indent=2)
