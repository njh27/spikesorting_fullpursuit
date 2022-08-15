from contextlib import nullcontext
import platform as sys_platform
import numpy as np
from numpy import uint8, ndarray, dtype
from numpy.compat import os_fspath, is_pathlib_path
from numpy.core.overrides import set_module

__all__ = ['memmap']

dtypedescr = dtype
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]

mode_equivalents = {
    "readonly":"r",
    "copyonwrite":"c",
    "readwrite":"r+",
    "write":"w+"
    }


@set_module('numpy')
class NumpyMemMapPrivate(ndarray):
    """Create a memory-map to an array stored in a *binary* file on disk.
    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.
    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.
    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.
    Flush the memmap instance to write the changes to the file. Currently there
    is no API to close the underlying ``mmap``. It is tricky to ensure the
    resource is actually closed, since it may be shared between different
    memmap instances.
    Parameters
    ----------
    filename : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:
        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+
        Default is 'r+'.
    offset : int, optional
        In the file, array data starts at this offset. Since `offset` is
        measured in bytes, it should normally be a multiple of the byte-size
        of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
        file are valid; The file will be extended to accommodate the
        additional data. By default, ``memmap`` will start at the beginning of
        the file, even if ``filename`` is a file pointer ``fp`` and
        ``fp.tell() != 0``.
    shape : tuple, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.
    order : {'C', 'F'}, optional
        Specify the order of the ndarray memory layout:
        :term:`row-major`, C-style or :term:`column-major`,
        Fortran-style.  This only has an effect if the shape is
        greater than 1-D.  The default order is 'C'.
    Attributes
    ----------
    filename : str or pathlib.Path instance
        Path to the mapped file.
    offset : int
        Offset position in the file.
    mode : str
        File mode.
    Methods
    -------
    flush
        Flush any changes in memory to file on disk.
        When you delete a memmap object, flush is called first to write
        changes to disk.
    See also
    --------
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.
    Notes
    -----
    The memmap object can be used anywhere an ndarray is accepted.
    Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
    ``True``.
    Memory-mapped files cannot be larger than 2GB on 32-bit systems.
    When a memmap causes a file to be created or extended beyond its
    current size in the filesystem, the contents of the new part are
    unspecified. On systems with POSIX filesystem semantics, the extended
    part will be filled with zero bytes.
    Examples
    --------
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))
    This example uses a temporary file so that doctest doesn't write
    files to your directory. You would use a 'normal' filename.
    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')
    Create a memmap with dtype and shape that matches our data:
    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]], dtype=float32)
    Write data to memmap array:
    >>> fp[:] = data[:]
    >>> fp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fp.filename == path.abspath(filename)
    True
    Flushes memory changes to disk in order to read them back
    >>> fp.flush()
    Load the memmap and verify data was stored:
    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> newfp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    Read-only memmap:
    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> fpr.flags.writeable
    False
    Copy-on-write memmap:
    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
    >>> fpc.flags.writeable
    True
    It's possible to assign to copy-on-write array, but values are only
    written into the memory copy of the array, and not written to disk:
    >>> fpc
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fpc[0,:] = 0
    >>> fpc
    memmap([[  0.,   0.,   0.,   0.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    File on disk is unchanged:
    >>> fpr
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    Offset into a memmap:
    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
    >>> fpo
    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)
    """

    __array_priority__ = -100.0

    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        # Import here to minimize 'import numpy' overhead
        import mmap
        import os.path
        try:
            mode = mode_equivalents[mode]
        except KeyError as e:
            if mode not in valid_filemodes:
                raise ValueError(
                    "mode must be one of {!r} (got {!r})"
                    .format(valid_filemodes + list(mode_equivalents.keys()), mode)
                ) from None

        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given")

        if hasattr(filename, 'read'):
            f_ctx = nullcontext(filename)
        else:
            f_ctx = open(os_fspath(filename), ('r' if mode == 'c' else mode)+'b')

        with f_ctx as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize

            if shape is None:
                bytes = flen - offset
                if bytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                            "multiple of the data-type size.")
                size = bytes // _dbytes
                shape = (size,)
            else:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                size = np.intp(1)  # avoid default choice of np.int_, which might overflow
                for k in shape:
                    size *= k

            bytes = int(offset + size*_dbytes)

            if mode in ('w+', 'r+') and flen < bytes:
                fid.seek(bytes - 1, 0)
                fid.write(b'\0')
                fid.flush()

            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            array_offset = offset - start
            if sys_platform.system() == 'Windows':
                # Flags only work on Posix so keep this as normal
                if mode == 'c':
                    acc = mmap.ACCESS_COPY
                elif mode == 'r':
                    acc = mmap.ACCESS_READ
                else:
                    acc = mmap.ACCESS_WRITE
                mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
            else:
                # Always attempt private, since that is purpose of this class
                MAP_flags = mmap.MAP_PRIVATE
                PROT = 0
                if mode == 'c':
                    # Need to check this attribute is available on system
                    if hasattr(mmap, "MAP_DENYWRITE"):
                        MAP_flags = MAP_flags | getattr(mmap, "MAP_DENYWRITE")
                    PROT = mmap.PROT_READ
                elif mode == 'r':
                    if hasattr(mmap, "MAP_DENYWRITE"):
                        MAP_flags = MAP_flags | getattr(mmap, "MAP_DENYWRITE")
                    PROT = mmap.PROT_READ
                else:
                    PROT = mmap.PROT_WRITE

                    MAP_flags = mmap.MAP_PRIVATE

                print("making map with flags", MAP_flags, "prot", PROT)
                mm = mmap.mmap(fid.fileno(), bytes, offset=start,
                                flags=MAP_flags, prot=PROT)

            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode

            if is_pathlib_path(filename):
                # special case - if we were constructed with a pathlib.path,
                # then filename is a path object, not a string
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                # py3 returns int for TemporaryFile().name
                self.filename = os.path.abspath(fid.name)
            # same as memmap copies (e.g. memmap + 1)
            else:
                self.filename = None

        return self

    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None

    def flush(self):
        """
        Write any changes in the array to the file on disk.
        For further information, see `memmap`.
        Parameters
        ----------
        None
        See Also
        --------
        memmap
        """
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()

    def __array_wrap__(self, arr, context=None):
        arr = super().__array_wrap__(arr, context)

        # Return a memmap if a memmap was given as the output of the
        # ufunc. Leave the arr class unchanged if self is not a memmap
        # to keep original memmap subclasses behavior
        if self is arr or type(self) is not memmap:
            return arr
        # Return scalar instead of 0d memmap, e.g. for np.sum with
        # axis=None
        if arr.shape == ():
            return arr[()]
        # Return ndarray otherwise
        return arr.view(np.ndarray)

    def __getitem__(self, index):
        res = super().__getitem__(index)
        if type(res) is NumpyMemMapPrivate and res._mmap is None:
            return res.view(type=ndarray)
        return res





class NumpyMemMapPrivateOLD(ndarray):
    """
    Create a memory-map to an array stored in a *binary* file on disk.
    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  Numpy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.
    Parameters
    ----------
    filename : str or file-like object
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:
        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+
        Default is 'r+'.
    offset : int, optional
        In the file, array data starts at this offset. Since `offset` is
        measured in bytes, it should be a multiple of the byte-size of
        `dtype`. Requires ``shape=None``. The default is 0.
    shape : tuple, optional
        The desired shape of the array. By default, the returned array will be
        1-D with the number of elements determined by file size and data-type.
    order : {'C', 'F'}, optional
        Specify the order of the ndarray memory layout: C (row-major) or
        Fortran (column-major).  This only has an effect if the shape is
        greater than 1-D.  The default order is 'C'.
    Methods
    -------
    close
        Close the memmap file.
    flush
        Flush any changes in memory to file on disk.
        When you delete a memmap object, flush is called first to write
        changes to disk before removing the object.
    Notes
    -----
    The memmap object can be used anywhere an ndarray is accepted.
    Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
    ``True``.
    Memory-mapped arrays use the Python memory-map object which
    (prior to Python 2.5) does not allow files to be larger than a
    certain size depending on the platform. This size is always < 2GB
    even on 64-bit systems.
    Examples
    --------
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))
    This example uses a temporary file so that doctest doesn't write
    files to your directory. You would use a 'normal' filename.
    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')
    Create a memmap with dtype and shape that matches our data:
    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.]], dtype=float32)
    Write data to memmap array:
    >>> fp[:] = data[:]
    >>> fp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    Deletion flushes memory changes to disk before removing the object:
    >>> del fp
    Load the memmap and verify data was stored:
    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> newfp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    Read-only memmap:
    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> fpr.flags.writeable
    False
    Copy-on-write memmap:
    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
    >>> fpc.flags.writeable
    True
    It's possible to assign to copy-on-write array, but values are only
    written into the memory copy of the array, and not written to disk:
    >>> fpc
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fpc[0,:] = 0
    >>> fpc
    memmap([[  0.,   0.,   0.,   0.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    File on disk is unchanged:
    >>> fpr
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    Offset into a memmap:
    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
    >>> fpo
    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)
    """
    __array_priority__ = -100.0
    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        # Import here to minimize 'import numpy' overhead
        import mmap
        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" % \
                                 (valid_filemodes + mode_equivalents.keys()))
        if hasattr(filename,'read'):
            fid = filename
        else:
            fid = open(filename, (mode == 'c' and 'r' or mode)+'b')
        if (mode == 'w+') and shape is None:
            raise(ValueError, "shape must be given")
        fid.seek(0, 2)
        flen = fid.tell()
        descr = dtypedescr(dtype)
        _dbytes = descr.itemsize
        if shape is None:
            nbytes = flen - offset
            if (nbytes % _dbytes):
                fid.close()
                raise(ValueError, "Size of available data is not a "\
                      "multiple of data-type size.")
            size = nbytes // _dbytes
            shape = (size,)
        else:
            if not isinstance(shape, tuple):
                shape = (shape,)
            size = 1
            for k in shape:
                size *= k
        nbytes = int(offset + size*_dbytes)
        if mode == 'w+' or (mode == 'r+' and flen < nbytes):
            fid.seek(nbytes - 1, 0)
            fid.write(bytearray([0]))
            fid.flush()

        if sys_platform.system() == 'Windows':
            # Flags only work on Posix so keep this as normal
            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE
            if sys.version_info[:2] >= (2,6):
                # The offset keyword in mmap.mmap needs Python >= 2.6
                start = offset - offset % mmap.ALLOCATIONGRANULARITY
                nbytes -= start
                offset -= start
                mm = mmap.mmap(fid.fileno(), nbytes, offset=start,
                                access=acc)
            else:
                mm = mmap.mmap(fid.fileno(), nbytes, access=acc)
        else:
            # Always attempt private, since that is purpose of this class
            MAP_flags = mmap.MAP_PRIVATE
            PROT = 0
            if mode == 'c':
                # Need to check this attribute is available on system
                if hasattr(mmap, "MAP_DENYWRITE"):
                    MAP_flags = MAP_flags | getattr(mmap, "MAP_DENYWRITE")
                PROT = mmap.PROT_READ
            elif mode == 'r':
                if hasattr(mmap, "MAP_DENYWRITE"):
                    MAP_flags = MAP_flags | getattr(mmap, "MAP_DENYWRITE")
                PROT = mmap.PROT_READ
            else:
                PROT = mmap.PROT_WRITE
            if sys.version_info[:2] >= (2,6):
                # The offset keyword in mmap.mmap needs Python >= 2.6
                start = offset - offset % mmap.ALLOCATIONGRANULARITY
                nbytes -= start
                offset -= start
                mm = mmap.mmap(fid.fileno(), nbytes, offset=start,
                                flags=MAP_flags, prot=PROT)
            else:
                mm = mmap.mmap(fid.fileno(), nbytes,
                                flags=MAP_flags, prot=PROT)


        self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
            offset=offset, order=order)
        self._mmap = mm
        return self
    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap'):
            self._mmap = obj._mmap
        else:
            self._mmap = None
    def flush(self):
        """
        Write any changes in the array to the file on disk.
        For further information, see `memmap`.
        Parameters
        ----------
        None
        See Also
        --------
        memmap
        """
        if self._mmap is not None:
            print("mmap flush")
            self._mmap.flush()
    def sync(self):
        """This method is deprecated, use `flush`."""
        warnings.warn("Use ``flush``.", DeprecationWarning)
        self.flush()
    def _close(self):
        """Close the memmap file.  Only do this when deleting the object."""
        if self.base is self._mmap:
            # The python mmap probably causes flush on close, but
            # we put this here for safety
            self._mmap.flush()
            self._mmap.close()
            self._mmap = None
    def close(self):
        """Close the memmap file. Does nothing."""
        warnings.warn("``close`` is deprecated on memmap arrays.  Use del",
                      DeprecationWarning)
    def __del__(self):
        # We first check if we are the owner of the mmap, rather than
        # a view, so deleting a view does not call _close
        # on the parent mmap
        print("attempting delete")
        if self._mmap is self.base:
            try:
                # First run tell() to see whether file is open
                self._mmap.tell()
            except ValueError:
                pass
            else:
                self._close()



# class shit(np.memmap):
#     def __new__(self, filename, dtype=np.float64, mode='r+', offset=0,
#                 shape=None, order='C', MADV_flags=None, MAP_flags=None):
#         """ Multiple flags can be input with "|", e.g.:
#             mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS
#         Note things like mmap.MAP_ANONYMOUS cannot be used with a specified file
#         name!
#         These values should be imported from mmap first and passed. MAP flags
#         do not exist on Windows and Windows is private by default.
#         """
#         # Import here to minimize 'import numpy' overhead
#         import mmap
#         if sys_platform.system() == 'Windows':
#             MAP_flags=None
#
#         descr = np.dtype(dtype)
#         if not isinstance(shape, tuple): shape = (shape,)
#         nbytes = descr.itemsize
#         for k in shape:
#             nbytes *= k
#
#         if mode == "r":
#             mmap_mode = "rb"
#         elif mode == "r+":
#             mmap_mode = "r+b"
#         elif mode == "w+":
#             mmap_mode = "r+b"
#         elif mode == "c":
#             mmap_mode = "r+b"
#         else:
#             raise ValueError("Unrecognized file open mode '{0}'.".format(mode))
#
#         self = np.memmap.__new__(self, filename, dtype=dtype, mode=mode, offset=offset,
#                                  shape=shape, order=order)
#         self._mmap.close()
#         del self._mmap
#         if MAP_flags is not None:
#             with open(filename, mmap_mode) as f:
#                 mm = mmap.mmap(f.fileno(), nbytes, flags=MAP_flags)
#         else:
#             with open(filename, mmap_mode) as f:
#                 mm = mmap.mmap(f.fileno(), nbytes)
#         self._mmap = mm
#         return self
#     # def __array_finalize__(self, obj):
#     #     if hasattr(obj, '_mmap'):
#     #         self._mmap = obj._mmap
#     #     else:
#     #         self._mmap = None
#     # def flush(self): pass
#     # def sync(self):
#     #     """This method is deprecated, use `flush`."""
#     #     warnings.warn("Use ``flush``.", DeprecationWarning)
#     #     self.flush()
#     def _close(self):
#         """Close the memmap file.  Only do this when deleting the object."""
#         if self.base is self._mmap:
#             # The python mmap probably causes flush on close, but
#             # we put this here for safety
#             print("FLUSH AND CLOSE!")
#             self._mmap.flush()
#             self._mmap.close()
#             self._mmap = None
#     def close(self):
#         """Close the memmap file. Does nothing."""
#         warnings.warn("``close`` is deprecated on memmap arrays.  Use del", DeprecationWarning)
#     def __del__(self):
#         # We first check if we are the owner of the mmap, rather than
#         # a view, so deleting a view does not call _close
#         # on the parent mmap
#         print("checking for delete")
#         if self._mmap is self.base:
#             try:
#                 # First run tell() to see whether file is open
#                 print("TRYING tell")
#                 self._mmap.tell()
#             except ValueError:
#                 pass
#             else:
#                 self._close()


# class NumpyMemMap(np.ndarray):
#     """
#     """
#     __array_priority__ = -100.0
#     def __new__(cls, filename, dtype=np.float64, mode='r+', offset=0,
#                 shape=None, order='C', MADV_flags=None, MAP_flags=None):
#         # Import here to minimize 'import numpy' overhead
#         import mmap
#         descr = np.dtype(dtype)
#         if not isinstance(shape, tuple): shape = (shape,)
#         nbytes = descr.itemsize
#         n_elem = 1
#         for k in shape:
#             n_elem *= k
#             nbytes *= k
#         if mode == "r":
#             mmap_mode = "rb"
#         elif mode == "r+":
#             mmap_mode = "r+b"
#         elif mode == "w+":
#             mmap_mode = "r+b"
#         elif mode == "c":
#             mmap_mode = "r+b"
#         else:
#             raise ValueError("Unrecognized file open mode '{0}'.".format(mode))
#
#         print("opening in mode", mmap_mode)
#         if MAP_flags is not None:
#             with open(filename, mmap_mode) as f:
#                 mm = mmap.mmap(f.fileno(), nbytes, flags=MAP_flags)
#         else:
#             with open(filename, mmap_mode) as f:
#                 mm = mmap.mmap(f.fileno(), nbytes)
#
#         self = np.memmap.__new__(cls, filename, dtype=dtype, mode=mode, offset=offset,
#                                  shape=shape, order=order)
#         self._mmap.close()
#         del self._mmap
#
#
#
#         self._mmap = mm
#         return self
#
#
#
#     def flush(self): pass
#     def sync(self):
#         """This method is deprecated, use `flush`."""
#         warnings.warn("Use ``flush``.", DeprecationWarning)
#         self.flush()
#     def _close(self):
#         """Close the memmap file.  Only do this when deleting the object."""
#         if self.base is self._mmap:
#             # The python mmap probably causes flush on close, but
#             # we put this here for safety
#             print("FLUSH AND CLOSE nd array!")
#             self._mmap.flush()
#             self._mmap.close()
#             self._mmap = None
#     def close(self):
#         """Close the memmap file. Does nothing."""
#         warnings.warn("``close`` is deprecated on memmap arrays.  Use del", DeprecationWarning)
#     def __del__(self):
#         # We first check if we are the owner of the mmap, rather than
#         # a view, so deleting a view does not call _close
#         # on the parent mmap
#         print("tring delete nd array")
#         if self._mmap is self.base:
#             try:
#                 # First run tell() to see whether file is open
#                 print("TESTING tell nd array")
#                 self._mmap.tell()
#             except ValueError:
#                 pass
#             else:
#                 self._close()



# class NumpyMemMap(np.ndarray):
#     """
#     """
#     __array_priority__ = -100.0
#     def __new__(cls, filename, dtype=np.float64, mode='r+', offset=0,
#                 shape=None, order='C', MADV_flags=None, MAP_flags=None):
#         # Import here to minimize 'import numpy' overhead
#         import mmap
#         descr = np.dtype(dtype)
#         if not isinstance(shape, tuple): shape = (shape,)
#         nbytes = descr.itemsize
#         n_elem = 1
#         for k in shape:
#             n_elem *= k
#             nbytes *= k
#         if mode == "r":
#             mmap_mode = "rb"
#         elif mode == "r+":
#             mmap_mode = "r+b"
#         elif mode == "w+":
#             # Need to overwrite/create file for w+
#             print("Overwriting the byes!")
#             s = struct.pack(np.ctypeslib.as_ctypes_type(descr)._type_*n_elem, *[0 for x in range(n_elem)])
#             with open(filename, "wb") as f:
#                 f.write(s)
#             mmap_mode = "r+b"
#         elif mode == "c":
#             mmap_mode = "r+b"
#         else:
#             raise ValueError("Unrecognized file open mode '{0}'.".format(mode))
#
#         print("opening in mode", mmap_mode)
#         if MAP_flags is not None:
#             f = open(filename, mmap_mode)
#             # with open(filename, mmap_mode) as f:
#             mm = mmap.mmap(f.fileno(), nbytes, flags=MAP_flags)
#         else:
#             f = open(filename, mmap_mode)
#             # with open(filename, mmap_mode) as f:
#             mm = mmap.mmap(f.fileno(), nbytes)
#         if mode == "w+":
#             self = np.ndarray.__new__(cls, shape, dtype=dtype, buffer=mm, order=order)
#         else:
#             self = np.frombuffer(mm, dtype=dtype, count=- 1, offset=offset)
#         self._mmap = mm
#         self.f = f
#         return self
#
#
#
#
#
#     def __array_finalize__(self, obj):
#         if hasattr(obj, '_mmap'):
#             self._mmap = obj._mmap
#         else:
#             self._mmap = None
#     # def flush(self): pass
#     def sync(self):
#         """This method is deprecated, use `flush`."""
#         warnings.warn("Use ``flush``.", DeprecationWarning)
#         self.flush()
#     def _close(self):
#         """Close the memmap file.  Only do this when deleting the object."""
#         if self.base is self._mmap:
#             # The python mmap probably causes flush on close, but
#             # we put this here for safety
#             print("FLUSH AND CLOSE nd array!")
#             self._mmap.flush()
#             self._mmap.close()
#             self._mmap = None
#             self.f.close()
#     def close(self):
#         """Close the memmap file. Does nothing."""
#         warnings.warn("``close`` is deprecated on memmap arrays.  Use del", DeprecationWarning)
#     def __del__(self):
#         # We first check if we are the owner of the mmap, rather than
#         # a view, so deleting a view does not call _close
#         # on the parent mmap
#         print("tring delete nd array")
#         if self._mmap is self.base:
#             try:
#                 # First run tell() to see whether file is open
#                 print("TESTING tell nd array")
#                 self._mmap.tell()
#             except ValueError:
#                 pass
#             else:
#                 self._close()
#                 self.f.close()
