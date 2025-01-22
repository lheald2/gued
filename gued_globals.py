### Global Variables for LUED data set
PATH_SEPARATOR = '\\' # input how folders are separator in the file path
# Variable for reading files
SEPARATORS = '_'

# Variables for Center Finding Algorithm
CENTER_GUESS = (465, 475)
RADIUS_GUESS = 35
DISK_RADIUS = 3
THRESHOLD = 0

# Variable for Generating Background
CORNER_RADIUS = 20
CHECK_NUMBER = 50

# Variables for Masking
MASK_CENTER = [525, 515]
MASK_RADIUS = 50
#ADDED_MASK = [[546, 470, 40]]
ADDED_MASK = []

# Used throughout code as the threshold for cutting out date. This is the default value but other values can be set for the functions using this variable
STD_FACTOR = 3

# Specifies the maximum number of workers to be used when running concurrent.futures
MAX_PROCESSORS = 6

# Adjust figure size 
FIGSIZE = (12,4)

# Path for Theory Package
PATH_DCS = '/packages/dcs_repositiory/3.7MeV/'

