import numpy as np


class NpViewsMatrix(object):
    """Allows construction of a matrix from numpy views into other matrices.
    Modification of this matrix in place will change the data of the original
    matrix! Class assumes that each row of NpViewsMatrix will be assigned as
    a view of another numpy array where each view has the same size property
    and will be flattened for reading/interpreting row-wise. Data must be
    assigned "row-wise" one view at a time. If individual items are assigned
    first, rows will be allocated with a numpy array of zeros.
    """
    def __init__(self, n_rows, n_cols, dtype=np.float64):
        self.n_cols = n_cols
        self.rows = [None for x in range(0, n_rows)]
        self.dtype = dtype

    def _setrow_(self, row, values):
        if isinstance(values, np.ndarray):
            if values.size == self.n_cols:
                self.rows[row] = values
        elif self.rows[row] is None:
            self.rows[row] = np.zeros(self.n_cols, dtype=self.dtype)
        self.rows[row].flat = values
        return None

    def __setitem__(self, inds, values):
        for i in inds:
            print(type(i), i)
        print(values)


        if isinstance(inds, int):
            self._setrow_(inds, values)
        if isinstance(inds, tuple):
            if len(inds) > 2:
                raise ValueError("NpViewsMatrix only supports 2D representations.")
            if isinstance(inds[0], slice):
                sl_step = 1 if inds[0].step is None else inds[0].step
                for x in range(inds[0].start, inds[0].stop, sl_step):
                    self.rows[x].flat[inds[1]] = values

            if self.rows[inds[0]] is None:
                raise ValueError("Row data not yet set for row {0}.".format(inds[0]))
            self.rows[inds[0]].flat[inds[1]] = values
