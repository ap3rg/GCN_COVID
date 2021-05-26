# From paper Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks
# https://arxiv.org/pdf/2007.03113.pdf

EMB_WINDOW = 5 # number of dates will be window + 1
PRED_WINDOW = 1

# Training
DROPOUT = 0.5
L2_REGULARIZATION = 5e-4
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_EPOCHS = 50

# Data
SOURCE_NAME = "start_poly_id"
TARGET_NAME = "end_poly_id"
NODE_NAME = "poly_id"
EDGE_WEIGHT_NAME = "movement"

# Debug
DEBUG = False

