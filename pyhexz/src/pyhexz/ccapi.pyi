# Hand-crafted .pyi interface file to make using the C++ bindings a bit nicer.

class CppMoveSuggester:
    def suggest_move(self, request_data: bytes) -> bytes:
        """Calls the MoveSuggester::SuggestMove C++ method. 
        
        Args:
            request_data: a byte-serialized hexz_pb2.SuggestMoveRequest
        Returns:
            a byte-serialized hexz_pb2.SuggestMoveResponse
        Raises:
            ValueError: if anything goes wrong
        """
        ...

    def __init__(self, path: str): 
        """Creates a C++ MoveSuggester wrapper.
        
        Args:
            path: Path to a PyTorch ScriptModule file.
        """
        ...
