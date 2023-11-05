from libcpp.string cimport string

cdef extern from "pyapi.h" namespace "hexz":
    cdef cppclass MoveSuggester:
        MoveSuggester() except +
        void LoadModel(string) except +
        string SuggestMove(string) except +
