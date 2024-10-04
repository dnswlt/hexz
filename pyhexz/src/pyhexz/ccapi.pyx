from libcpp.string cimport string

from pyhexz.hexz_pyapi cimport MoveSuggester

cdef class CppMoveSuggester:
    cdef MoveSuggester _ms

    def __init__(self, str path):
        cdef string path_utf8
        path_utf8 = path.encode("utf-8")

        self._ms = MoveSuggester()
        self._ms.LoadModel(path_utf8)
    
    def suggest_move(self, string input):
        cdef string result
        result = self._ms.SuggestMove(input)
        return result
