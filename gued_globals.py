<<<<<<< HEAD
### Global Variables for s1 data set
=======
"""Python file for establishing global variables that change from experiment to experiment. Each global variable should be notated with 
all caps."""
>>>>>>> main

### Global Variables for demo data set
# Variable for reading files
<<<<<<< HEAD
SEPARATORS = '_'
=======
PATH_SEPARATOR = '/' # input how folders are separator in the file path
SEPARATORS = '_' # underscore or dash usually, based on how the files are named
>>>>>>> main

# Variables for Center Finding Algorithm
CENTER_GUESS = (465, 475)
RADIUS_GUESS = 35
DISK_RADIUS = 3 
THRESHOLD = 1000

# Variable for Generating Background
CORNER_RADIUS = 20
CHECK_NUMBER = 50

# Variables for Masking
<<<<<<< HEAD
MASK_CENTER = [525, 520]
MASK_RADIUS = 50
ADDED_MASK = []
# ADDED_MASK = [
#     [432, 464, 16],
#     [445, 440, 35], 
#     [471, 427, 22],
#     [476, 482, 35]]
=======
MASK_CENTER = [525, 515] # x, y position for mask
MASK_RADIUS = 50 # radius for mask
#ADDED_MASK = [[546, 470, 40]]
ADDED_MASK = []
>>>>>>> main

# Used throughout code as the threshold for cutting out date. This is the default value but other values can be set for the functions using this variable
STD_FACTOR = 3

# Specifies the maximum number of workers to be used when running concurrent.futures
MAX_PROCESSORS = 8

# Adjust figure size 
FIGSIZE = (10,4)

# Path for Theory Package
PATH_DCS = '/packages/dcs_repositiory/3.7MeV/'
