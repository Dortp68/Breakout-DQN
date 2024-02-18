from prerocess import *
BATCH_SIZE = 32
LR = 0.00025
GAMMA = 0.99
MEMORY_CAPACITY = 300000
MEMORY_START_SIZE = 50000
MEMORY_START_SIZE = MEMORY_START_SIZE if MEMORY_START_SIZE > BATCH_SIZE else BATCH_SIZE
Q_NETWORK_ITERATION = 5000
MAX_STEPS = 10000
EPSILON_GREEDY_FRAMES = 1000000.0

env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array", obs_type="grayscale")
env = PreprocessAtari(env)
env = FrameBuffer(env, n_frames=4)
NUM_ACTIONS = env.action_space.n
SHAPE_STATES = env.observation_space.shape
NUM_STATES = SHAPE_STATES[0]*SHAPE_STATES[1]*SHAPE_STATES[2]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
INP_SHAPE = (BATCH_SIZE,)+SHAPE_STATES