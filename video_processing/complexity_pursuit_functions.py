import numpy as np


def return_mask(half_life, half_lives_per_mask, max_mask_lenght):
    mask_lenght = max(half_lives_per_mask*half_life, max_mask_lenght)
    print('Retornando máscara para Complexity Pursuit com tamanho: ', mask_lenght)
    mask = (2**(1/half_life))**(np.arange(1, mask_lenght))
    mask = mask/(np.sum(np.abs(mask[1:])))
    mask[0] = -1
    return mask
