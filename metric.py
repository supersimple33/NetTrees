"""
Defines classes to compute the distance between points in any general metric spaces.
"""
from abc import ABC, abstractmethod
import os
from arithmetic import FloatArithmetic

try:
    import numpy as np
except ImportError:
    np = None

try:
    from numba import njit
except ImportError:
    njit = None

from config import config


NUMBA_AVAILABLE = np is not None and njit is not None
NUMBA_DISABLED = os.getenv("NETTREES_DISABLE_NUMBA", "0") == "1"


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _euclidean_distance_numba(first, second):
        total = 0.0
        for i in range(first.shape[0]):
            diff = first[i] - second[i]
            total += diff * diff
        return total ** 0.5


    @njit(cache=True, fastmath=True)
    def _manhattan_distance_numba(first, second):
        total = 0.0
        for i in range(first.shape[0]):
            total += abs(first[i] - second[i])
        return total


    @njit(cache=True, fastmath=True)
    def _linfinity_distance_numba(first, second):
        result = 0.0
        for i in range(first.shape[0]):
            current = abs(first[i] - second[i])
            if current > result:
                result = current
        return result


def _cache_key(first, second):
    first_id = id(first)
    second_id = id(second)
    return (first_id, second_id) if first_id <= second_id else (second_id, first_id)


def _numba_coords(point):
    if not hasattr(point, "as_float_array"):
        return None
    return point.as_float_array()

class Metric(ABC):
    """
    An abstract base class for any arbitrary metric which delegates the distance 
    computation to its concrete subclasses.

    Parameters
    ----------
    cachedist : bool
        Determines whether the computed distances should be stored in 
        a dictionary to avoid recalculations.
    """
    def __init__(self, cachedist, use_numba=True):
        self.cachedist = cachedist
        self.use_numba = bool(
            use_numba and
            NUMBA_AVAILABLE and
            not NUMBA_DISABLED and
            isinstance(config.arithmatic, FloatArithmetic)
        )
        self.distdict = {}
        self.reset()
        
    def reset(self):
        """
        Resets the counter tracing the number of distance computations 
        and clears the dictionary storing the computed distances.
        
        Parameters
        ----------
        None
        
        Returns:
        -------
        None
        """
        self.counter = 0
        self.distdict.clear()
        
    def dist(self, first, *others):
        """
        Computes the minimum distance of a point to other points.
        
        Parameters
        ----------
        first : Point
            The first point.
        others: variable length argument list of type Point
            A collection of points.
        
        Returns:
        -------
        float or Decimal
            The minimum distance.
        """
        if len(others) == 0: 
            raise TypeError("Metric.dist: this method should have at least two arguments")
        min_distance = self.getdist(first, others[0])
        for other in others[1:]:
            current_distance = self.getdist(first, other)
            if current_distance < min_distance:
                min_distance = current_distance
        return min_distance
    
    def getdist(self, first, second):
        """
        Computes the distance between two points.
        
        Parameters
        ----------
        first : Point
            The first point.
        second: Point
            The second point.
            
        Returns:
        -------
        float or Decimal
            The distance between `first` and `second`.
        """
        key = _cache_key(first, second) if self.cachedist else None
        if self.cachedist:
            cached_distance = self.distdict.get(key)
            if cached_distance is not None:
                return cached_distance

        distance = 0
        if first != second:
            distance = self.distance(first, second)
            self.counter += 1

        if self.cachedist:
            self.distdict[key] = distance
        return distance
    
    @abstractmethod
    def distance(self, first, second):
        """
        Returns the distance between two points in a certain metric.
        To be implemented by concerete metric subclasses.
        
        Parameters:
        ----------
        first : Point
            The first point.
        second : Point
            The second point.
        
        Returns:
        -------
        float or Decimal
            The distance between the first and the second points.
        """
        pass
    
    def __str__(self):
        """
        Creates a string representation for a metric object.
        
        Parameters:
        ----------
        None
        
        Returns:
        -------
        str
            The class name.
        """
        return type(self).__name__

class Euclidean(Metric):
    def __init__(self, cachedist=False, use_numba=True):
        Metric.__init__(self, cachedist, use_numba)

    def distance(self, first, second):
        if self.use_numba:
            first_coords = _numba_coords(first)
            second_coords = _numba_coords(second)
            if first_coords is not None and second_coords is not None and first_coords.shape == second_coords.shape:
                return _euclidean_distance_numba(first_coords, second_coords)

        total = 0
        for i in range(len(first.coords)):
            diff = first[i] - second[i]
            total += diff ** 2
        return config.arithmatic.sqrt(total)


class Manhattan(Metric):
    def __init__(self, cachedist=False, use_numba=True):
        Metric.__init__(self, cachedist, use_numba)

    def distance(self, first, second):
        if self.use_numba:
            first_coords = _numba_coords(first)
            second_coords = _numba_coords(second)
            if first_coords is not None and second_coords is not None and first_coords.shape == second_coords.shape:
                return _manhattan_distance_numba(first_coords, second_coords)

        total = 0
        for i in range(len(first.coords)):
            total += abs(first[i] - second[i])
        return total

        
class LInfinity(Metric):
    def __init__(self, cachedist=False, use_numba=True):
        Metric.__init__(self, cachedist, use_numba)

    def distance(self, first, second):
        if self.use_numba:
            first_coords = _numba_coords(first)
            second_coords = _numba_coords(second)
            if first_coords is not None and second_coords is not None and first_coords.shape == second_coords.shape:
                return _linfinity_distance_numba(first_coords, second_coords)

        result = 0
        for i in range(len(first.coords)):
            current = abs(first[i] - second[i])
            if current > result:
                result = current
        return result

