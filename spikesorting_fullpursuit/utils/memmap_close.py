import numpy as np



class MemMapClose(np.memmap):
    """ Just a sublcass of numpy memmap that closes the memmap file upon
    deletion after flushing data. This stops Windows errors complaining
    about the file being used by another process and inaccessible. However,
    if multiple references exist, further attempts to access the closed
    memmap file by other objects will CRASH THE SYSTEM! """
    def __new__(subtype, filename, dtype=np.uint8, mode='r+', offset=0,
                shape=None, order='C'):
        return super().__new__(subtype, filename, dtype=dtype, mode=mode,
                                offset=offset, shape=shape, order=order)

    def __del__(self):
        if self._mmap is self.base:
            try:
                # First run tell() to see whether file is open
                self._mmap.tell()
            except ValueError:
                pass
            else:
                self.flush()
                self._mmap.close()

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)

    def __array_wrap__(self, arr, context=None):
        return super().__array_wrap__(arr, context=None)

    def __getitem__(self, index):
        return super().__getitem__(index)
