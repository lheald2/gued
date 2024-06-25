### Global Variables for entire code

# Variable for reading files
#SEPARATORS = ['-', '_'] # older data sets might have multiple separator values so use a list of strings
SEPARATORS = '_'    # newer data sets that only have one separator use a string not a list

# Variables for Center Finding Algorithm
CENTER_GUESS = (500, 500)
RADIUS_GUESS = 40
DISK_RADIUS = 3

# Variables for Generating Background
CORNER_RADIUS = 20
CHECK_NUMBER = 50

# Variables for Masking
MASK_CENTER = [520, 510]
MASK_RADIUS = 50
ADDED_MASK = []

# Used throughout code as the threshold for cutting out date. This is the default value but other values can be set for the functions using
# std_factor = 4
STD_FACTOR = 3

# Specifies the maximum number of workers to be used when running concurrent.futures
MAX_PROCESSORS = 6

# Path for Theory Package

PATH_DCS = 'C:\\Users\\laure\\OneDrive - University of Nebraska-Lincoln\\Documents\\Centurion Lab\\Coding Lab Notebook\\gued_package\\GUED_Analysis\\packages\\dcs_repositiory\\3.7MeV\\'
#path_dcs = '/sdf/home/l/lheald2/GUED/jupyter_notebook/user_notebooks/dcs_repository/3.7MeV/'
