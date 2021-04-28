import numpy as np


def return_mask(half_life, half_lives_per_mask, max_mask_lenght):
    mask_lenght = min(half_lives_per_mask*half_life, max_mask_lenght)
    print('Returning mask for Complexity Pursuit with size: ', mask_lenght)
    mask = (2**(-1/half_life))**np.arange(0, mask_lenght).T
    mask[0] = 0
    mask = mask/(np.sum(np.abs(mask)))
    mask[0] = -1
    return mask
