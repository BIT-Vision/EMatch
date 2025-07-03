import numpy as np


def events_to_voxel(events: np.ndarray, num_bins:int, height:int, width:int, pos:int=0, normalize=False, standardize=True) -> np.ndarray:
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] array containing one event per row in the form: [x, y, timestamp, polarity=(+1,-1)]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    :param pos: filter the polarity of events
    :return voxel: [B,H,W]
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)
    assert (pos == 0 or pos == 1 or pos == -1)

    # Inital voxel
    voxel_grid = np.zeros((num_bins, height, width), np.float32)
    voxel_grid = voxel_grid.ravel()  # stretch to one-dimensional array

    # Extract information
    ts = events[:, 2]
    xs = events[:, 0]
    ys = events[:, 1]
    pols = events[:, 3]

    # Normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = ts[-1]
    first_stamp = ts[0]
    deltaT = 0 if last_stamp == first_stamp else last_stamp - first_stamp
    ts = (num_bins - 1) * (ts - first_stamp) / deltaT
    
    # Discretize t to integer fields
    x0 = xs.astype(np.int64)
    y0 = ys.astype(np.int64)
    t0 = ts.astype(np.int64)
    vals = pols
    
    for xlim in [x0,x0+1]:
        for ylim in [y0,y0+1]:
            for tlim in [t0,t0+1]:
                mask = (xlim < width) & (xlim >= 0) & (ylim < height) & (ylim >= 0) & (tlim >= 0) & (tlim < num_bins)
                if pos > 0:
                    mask = np.logical_and(mask, pols > 0)
                elif pos < 0:
                    mask = np.logical_and(mask, pols < 0)

                # Assign events to coordinates
                interp_weights = vals * (1 - np.abs(xlim-xs)) * (1 - np.abs(ylim-ys)) * (1 - np.abs(tlim - ts))
                index = xlim.astype(np.int64) + ylim.astype(np.int64) * width + tlim.astype(np.int64) * width * height
                np.add.at(voxel_grid, index[mask], interp_weights[mask])

    if normalize:
        # _range = np.max(voxel_grid) - np.min(voxel_grid)
        # voxel_grid = (voxel_grid - np.min(voxel_grid)) / _range
        _range = np.max(abs(voxel_grid))
        voxel_grid = voxel_grid / _range

    if standardize:
        mask = np.nonzero(voxel_grid)
        if len(mask[0]):
            mean = np.mean(voxel_grid[mask])
            std = np.std(voxel_grid[mask])
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid
