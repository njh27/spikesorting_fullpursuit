import numpy as np
import platform as sys_platform



class NumpyMemMap(np.memmap):
    def __new__(self, filename, dtype=np.float64, mode='r+', offset=0,
                shape=None, order='C', MADV_flags=None, MAP_flags=None):
        """ Multiple flags can be input with "|", e.g.:
            mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS
        Note things like mmap.MAP_ANONYMOUS cannot be used with a specified file
        name!
        These values should be imported from mmap first and passed. MAP flags
        do not exist on Windows and Windows is private by default.
        """
        # Import here to minimize 'import numpy' overhead
        import mmap
        if sys_platform.system() == 'Windows':
            MAP_flags=None

        descr = np.dtype(dtype)
        if not isinstance(shape, tuple): shape = (shape,)
        bytes = descr.itemsize
        for k in shape:
            bytes *= k

        self = np.memmap.__new__(self, filename, dtype=dtype, mode=mode, offset=offset,
                                 shape=shape, order=order)
        self._mmap.close()
        if MAP_flags is not None:
            # Open read only binary
            with open(filename, "rb") as f:
                mm = mmap.mmap(f.fileno(), bytes, flags=MAP_flags)
        else:
            # Open read only binary
            with open(filename, "rb") as f:
                mm = mmap.mmap(f.fileno(), bytes)
        self._mmap = mm
        return self
    # def __array_finalize__(self, obj):
    #     if hasattr(obj, '_mmap'):
    #         self._mmap = obj._mmap
    #     else:
    #         self._mmap = None
    # def flush(self): pass
    # def sync(self):
    #     """This method is deprecated, use `flush`."""
    #     warnings.warn("Use ``flush``.", DeprecationWarning)
    #     self.flush()
    # def _close(self):
    #     """Close the memmap file.  Only do this when deleting the object."""
    #     if self.base is self._mmap:
    #         # The python mmap probably causes flush on close, but
    #         # we put this here for safety
    #         self._mmap.flush()
    #         self._mmap.close()
    #         self._mmap = None
    # def close(self):
    #     """Close the memmap file. Does nothing."""
    #     warnings.warn("``close`` is deprecated on memmap arrays.  Use del", DeprecationWarning)
    # def __del__(self):
    #     # We first check if we are the owner of the mmap, rather than
    #     # a view, so deleting a view does not call _close
    #     # on the parent mmap
    #     if self._mmap is self.base:
    #         try:
    #             # First run tell() to see whether file is open
    #             self._mmap.tell()
    #         except ValueError:
    #             pass
    #         else:
    #             self._close()
