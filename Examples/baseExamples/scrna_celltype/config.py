SEEDS         = [42, 123, 456]
N_EPOCHS_BASE = 60
N_EPOCHS_PAI  = 150
LR            = 1e-3
BATCH_SIZE    = 128
TRAIN_FRACTION = 0.10  # 10% subsample — stress-tests rare cell type generalisation

N_CHUNKS = 50
D_MODEL  = 256
N_HEAD   = 4
N_LAYERS = 4
D_FF     = 512
DROPOUT  = 0.1

# Half-width variant used in the compression experiment
D_MODEL_SMALL = 128
D_FF_SMALL    = 256

FIGSHARE_ARTICLE = 12420968
PANCREAS_FILE    = 'human_pancreas_norm_complexBatch.h5ad'
N_TOP_GENES      = 2000
