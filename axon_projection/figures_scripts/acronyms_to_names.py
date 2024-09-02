"""Handy module to convert acronyms to their full names, and output to tex table."""
import ast

import pandas as pd

# Path to your CSV file
csv_file_path = "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/"
"axon_projection/out_a_p_12_obp_atlas/region_names_df.csv"

# Provided set of acronyms
acronyms_set = [
    "fiber tracts",
    "ec",
    "fp",
    "cpd",
    "or",
    "cing",
    "alv",
    "dhc",
    "fr",
    "hbc",
    "scwm",
    "P",
    "HY",
    "ZI",
    "PH",
    "DMH",
    "PVi",
    "AV",
    "IAD",
    "IAM",
    "LD",
    "LH",
    "MH",
    "IGL",
    "CL",
    "Eth",
    "LP",
    "PO",
    "MD",
    "PT",
    "PVT",
    "RT",
    "LGd",
    "ec",
    "SPFm",
    "VAL",
    "SPFm",
    "VAL",
    "CA",
    "VP",
    "VP",
    "or",
    "TH",
    "MB",
    "TH",
    "MB",
    "alv",
    "PRC",
    "NPC",
    "PRC",
    "NPC",
    "GPi",
    "GPi",
    "dhc",
    "CA",
    "DG",
    "CA",
    "DG",
    "fp",
    "HPF",
    "APr",
    "HPF",
    "APr",
    "ENT",
    "POST",
    "ENT",
    "POST",
    "ProS",
    "SUB",
    "ProS",
    "SUB",
    "ECT6a",
    "PERI6a",
    "TEa6b",
    "VISa",
    "VISrl",
    "VISa",
    "VISrl",
    "RSPagl",
    "RSPagl",
    "RSPd",
    "RSPd",
    "RSPv",
    "RSPv",
    "ProS",
    "RSPv",
    "TEa6a",
    "VISam",
    "TEa6b",
    "VISam",
    "TEa6a",
    "VISl",
    "VISli",
    "VISp",
    "VISl",
    "VISli",
    "VISp",
    "SUB",
    "VISpl",
    "VISpl",
    "VISpm",
    "VISpm",
    "VISpor",
    "VISpor",
    "VISpor",
    "root",
    "V3",
    "bic",
    "fp",
    "or",
    "alv",
    "fx",
    "dhc",
    "st",
    "sm",
    "root",
    "HY",
    "LHA",
    "ZI",
    "LM",
    "MM",
    "TM",
    "AD",
    "AM",
    "AV",
    "LD",
    "LGv",
    "PT",
    "RT",
    "LGd",
    "MG",
    "VAL",
    "VP",
    "TH",
    "PAL",
    "BST",
    "CA",
    "DG",
    "HPF",
    "ENT",
    "PAR",
    "POST",
    "PRE",
    "ProS",
    "SUB",
    "PIR",
    "MOp",
    "MOs",
    "SSp",
    "SSs",
    "VISC",
    "AUD",
    "PERI",
    "ECT",
    "GU",
    "ORB",
    "ACA",
    "RSP",
    "FRP",
    "PL",
    "ILA",
    "TEa",
    "MOp1",
    "MOp2",
    "MOp3",
    "MOp5",
    "MOp6a",
    "MOp6b",
    "MOp",
    "MOs",
    "SSp",
    "SSs",
    "CP",
    "PG",
    "SPVI",
    "Isocortex",
    "OLF",
    "CTXsp",
    "STR",
    "PAL",
    "TH",
    "HY",
    "MB",
    "PG",
    "MY",
    "MOs2",
    "MOp6a",
    "SSp-m4",
    "MOs6a",
    "MOs5",
    "SSs6a",
    "ACAd5",
    "SSp-ul5",
    "ORBl6a",
    "SSp-m6a",
    "MOs3",
    "SSp-n5",
    "MOp5",
    "RSPv5",
    "MOp3",
    "SSp-m5",
    "ACAd6a",
    "SSs5",
    "SSp-m3",
    "MOp2",
    "MOp1",
    "MOs1",
]

# Dictionary to store acronym to name mappings
acronym_name_mapping = {}


# Function to match acronyms and get their names
def find_acronym_names(acronyms_set, csv_file_path):
    """Find the acronym names in the provided CSV file and return a dictionary."""
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Dictionary to store acronym to name mappings
    acronym_name_mapping = {}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Convert the 'acronyms' and 'names' columns from string to lists
        acronyms_list = ast.literal_eval(row["acronyms"])
        names_list = ast.literal_eval(row["names"])

        # Create a mapping of each acronym to its corresponding name
        for acronym, name in zip(acronyms_list, names_list):
            acronym_name_mapping[acronym] = name

    # Search for the names corresponding to the acronyms in the provided set
    result = {acronym: acronym_name_mapping.get(acronym) for acronym in acronyms_set}

    return result


# Find and print the acronym names
acronym_names = find_acronym_names(acronyms_set, csv_file_path)
# sort the acronym names dict by alphabetical order
acronym_names = dict(sorted(acronym_names.items()))
# make a df of the dict
df = pd.DataFrame.from_dict(acronym_names, orient="index")
df.index.name = "Acronym"
df.columns = ["Name"]
# output it in latex
df.to_latex("acronyms_to_names.tex")
