from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Board(_message.Message):
    __slots__ = ["turn", "move", "last_revealed", "flat_fields", "score", "resources", "state"]
    class GameState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        INITIAL: _ClassVar[Board.GameState]
        RUNNING: _ClassVar[Board.GameState]
        FINISHED: _ClassVar[Board.GameState]
    INITIAL: Board.GameState
    RUNNING: Board.GameState
    FINISHED: Board.GameState
    TURN_FIELD_NUMBER: _ClassVar[int]
    MOVE_FIELD_NUMBER: _ClassVar[int]
    LAST_REVEALED_FIELD_NUMBER: _ClassVar[int]
    FLAT_FIELDS_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    turn: int
    move: int
    last_revealed: int
    flat_fields: _containers.RepeatedCompositeFieldContainer[Field]
    score: _containers.RepeatedScalarFieldContainer[int]
    resources: _containers.RepeatedCompositeFieldContainer[ResourceInfo]
    state: Board.GameState
    def __init__(self, turn: _Optional[int] = ..., move: _Optional[int] = ..., last_revealed: _Optional[int] = ..., flat_fields: _Optional[_Iterable[_Union[Field, _Mapping]]] = ..., score: _Optional[_Iterable[int]] = ..., resources: _Optional[_Iterable[_Union[ResourceInfo, _Mapping]]] = ..., state: _Optional[_Union[Board.GameState, str]] = ...) -> None: ...

class Field(_message.Message):
    __slots__ = ["type", "owner", "hidden", "value", "blocked", "lifetime", "next_val"]
    class CellType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        NORMAL: _ClassVar[Field.CellType]
        DEAD: _ClassVar[Field.CellType]
        GRASS: _ClassVar[Field.CellType]
        ROCK: _ClassVar[Field.CellType]
        FIRE: _ClassVar[Field.CellType]
        FLAG: _ClassVar[Field.CellType]
        PEST: _ClassVar[Field.CellType]
        DEATH: _ClassVar[Field.CellType]
    NORMAL: Field.CellType
    DEAD: Field.CellType
    GRASS: Field.CellType
    ROCK: Field.CellType
    FIRE: Field.CellType
    FLAG: Field.CellType
    PEST: Field.CellType
    DEATH: Field.CellType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    OWNER_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    BLOCKED_FIELD_NUMBER: _ClassVar[int]
    LIFETIME_FIELD_NUMBER: _ClassVar[int]
    NEXT_VAL_FIELD_NUMBER: _ClassVar[int]
    type: Field.CellType
    owner: int
    hidden: bool
    value: int
    blocked: int
    lifetime: int
    next_val: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, type: _Optional[_Union[Field.CellType, str]] = ..., owner: _Optional[int] = ..., hidden: bool = ..., value: _Optional[int] = ..., blocked: _Optional[int] = ..., lifetime: _Optional[int] = ..., next_val: _Optional[_Iterable[int]] = ...) -> None: ...

class ResourceInfo(_message.Message):
    __slots__ = ["num_pieces"]
    NUM_PIECES_FIELD_NUMBER: _ClassVar[int]
    num_pieces: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, num_pieces: _Optional[_Iterable[int]] = ...) -> None: ...

class Player(_message.Message):
    __slots__ = ["id", "name"]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...

class GameInfo(_message.Message):
    __slots__ = ["id", "host", "started", "type", "cpu_player"]
    ID_FIELD_NUMBER: _ClassVar[int]
    HOST_FIELD_NUMBER: _ClassVar[int]
    STARTED_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CPU_PLAYER_FIELD_NUMBER: _ClassVar[int]
    id: str
    host: str
    started: _timestamp_pb2.Timestamp
    type: str
    cpu_player: bool
    def __init__(self, id: _Optional[str] = ..., host: _Optional[str] = ..., started: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., type: _Optional[str] = ..., cpu_player: bool = ...) -> None: ...

class GameState(_message.Message):
    __slots__ = ["game_info", "seqnum", "modified", "players", "engine_state"]
    GAME_INFO_FIELD_NUMBER: _ClassVar[int]
    SEQNUM_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_FIELD_NUMBER: _ClassVar[int]
    PLAYERS_FIELD_NUMBER: _ClassVar[int]
    ENGINE_STATE_FIELD_NUMBER: _ClassVar[int]
    game_info: GameInfo
    seqnum: int
    modified: _timestamp_pb2.Timestamp
    players: _containers.RepeatedCompositeFieldContainer[Player]
    engine_state: GameEngineState
    def __init__(self, game_info: _Optional[_Union[GameInfo, _Mapping]] = ..., seqnum: _Optional[int] = ..., modified: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., players: _Optional[_Iterable[_Union[Player, _Mapping]]] = ..., engine_state: _Optional[_Union[GameEngineState, _Mapping]] = ...) -> None: ...

class GameEngineState(_message.Message):
    __slots__ = ["flagz", "classic", "freeform"]
    FLAGZ_FIELD_NUMBER: _ClassVar[int]
    CLASSIC_FIELD_NUMBER: _ClassVar[int]
    FREEFORM_FIELD_NUMBER: _ClassVar[int]
    flagz: GameEngineFlagzState
    classic: GameEngineClassicState
    freeform: GameEngineFreeformState
    def __init__(self, flagz: _Optional[_Union[GameEngineFlagzState, _Mapping]] = ..., classic: _Optional[_Union[GameEngineClassicState, _Mapping]] = ..., freeform: _Optional[_Union[GameEngineFreeformState, _Mapping]] = ...) -> None: ...

class GameEngineFlagzState(_message.Message):
    __slots__ = ["board", "free_cells", "normal_moves", "moves"]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    FREE_CELLS_FIELD_NUMBER: _ClassVar[int]
    NORMAL_MOVES_FIELD_NUMBER: _ClassVar[int]
    MOVES_FIELD_NUMBER: _ClassVar[int]
    board: Board
    free_cells: int
    normal_moves: _containers.RepeatedScalarFieldContainer[int]
    moves: _containers.RepeatedCompositeFieldContainer[GameEngineMove]
    def __init__(self, board: _Optional[_Union[Board, _Mapping]] = ..., free_cells: _Optional[int] = ..., normal_moves: _Optional[_Iterable[int]] = ..., moves: _Optional[_Iterable[_Union[GameEngineMove, _Mapping]]] = ...) -> None: ...

class GameEngineClassicState(_message.Message):
    __slots__ = ["board"]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    board: Board
    def __init__(self, board: _Optional[_Union[Board, _Mapping]] = ...) -> None: ...

class GameEngineFreeformState(_message.Message):
    __slots__ = ["board"]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    board: Board
    def __init__(self, board: _Optional[_Union[Board, _Mapping]] = ...) -> None: ...

class GameEngineMove(_message.Message):
    __slots__ = ["player_num", "move", "row", "col", "cell_type"]
    PLAYER_NUM_FIELD_NUMBER: _ClassVar[int]
    MOVE_FIELD_NUMBER: _ClassVar[int]
    ROW_FIELD_NUMBER: _ClassVar[int]
    COL_FIELD_NUMBER: _ClassVar[int]
    CELL_TYPE_FIELD_NUMBER: _ClassVar[int]
    player_num: int
    move: int
    row: int
    col: int
    cell_type: Field.CellType
    def __init__(self, player_num: _Optional[int] = ..., move: _Optional[int] = ..., row: _Optional[int] = ..., col: _Optional[int] = ..., cell_type: _Optional[_Union[Field.CellType, str]] = ...) -> None: ...

class MCTSExample(_message.Message):
    __slots__ = ["game_id", "board", "result", "move_stats"]
    class MoveStats(_message.Message):
        __slots__ = ["move", "visits", "win_rate"]
        MOVE_FIELD_NUMBER: _ClassVar[int]
        VISITS_FIELD_NUMBER: _ClassVar[int]
        WIN_RATE_FIELD_NUMBER: _ClassVar[int]
        move: GameEngineMove
        visits: int
        win_rate: float
        def __init__(self, move: _Optional[_Union[GameEngineMove, _Mapping]] = ..., visits: _Optional[int] = ..., win_rate: _Optional[float] = ...) -> None: ...
    GAME_ID_FIELD_NUMBER: _ClassVar[int]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    MOVE_STATS_FIELD_NUMBER: _ClassVar[int]
    game_id: str
    board: Board
    result: _containers.RepeatedScalarFieldContainer[int]
    move_stats: _containers.RepeatedCompositeFieldContainer[MCTSExample.MoveStats]
    def __init__(self, game_id: _Optional[str] = ..., board: _Optional[_Union[Board, _Mapping]] = ..., result: _Optional[_Iterable[int]] = ..., move_stats: _Optional[_Iterable[_Union[MCTSExample.MoveStats, _Mapping]]] = ...) -> None: ...

class ModelKey(_message.Message):
    __slots__ = ["name", "checkpoint"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    name: str
    checkpoint: int
    def __init__(self, name: _Optional[str] = ..., checkpoint: _Optional[int] = ...) -> None: ...

class AddTrainingExamplesRequest(_message.Message):
    __slots__ = ["model_key", "examples"]
    MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    model_key: ModelKey
    examples: _containers.RepeatedCompositeFieldContainer[TrainingExample]
    def __init__(self, model_key: _Optional[_Union[ModelKey, _Mapping]] = ..., examples: _Optional[_Iterable[_Union[TrainingExample, _Mapping]]] = ...) -> None: ...

class AddTrainingExamplesResponse(_message.Message):
    __slots__ = ["status", "latest_model", "error_message"]
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        STATUS_UNSPECIFIED: _ClassVar[AddTrainingExamplesResponse.Status]
        ACCEPTED: _ClassVar[AddTrainingExamplesResponse.Status]
        REJECTED_WRONG_MODEL: _ClassVar[AddTrainingExamplesResponse.Status]
        REJECTED_AT_CAPACITY: _ClassVar[AddTrainingExamplesResponse.Status]
        REJECTED_OTHER: _ClassVar[AddTrainingExamplesResponse.Status]
    STATUS_UNSPECIFIED: AddTrainingExamplesResponse.Status
    ACCEPTED: AddTrainingExamplesResponse.Status
    REJECTED_WRONG_MODEL: AddTrainingExamplesResponse.Status
    REJECTED_AT_CAPACITY: AddTrainingExamplesResponse.Status
    REJECTED_OTHER: AddTrainingExamplesResponse.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LATEST_MODEL_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    status: AddTrainingExamplesResponse.Status
    latest_model: ModelKey
    error_message: str
    def __init__(self, status: _Optional[_Union[AddTrainingExamplesResponse.Status, str]] = ..., latest_model: _Optional[_Union[ModelKey, _Mapping]] = ..., error_message: _Optional[str] = ...) -> None: ...

class TrainingExample(_message.Message):
    __slots__ = ["unix_micros", "turn", "encoding", "board", "move_probs", "result", "stats"]
    class Encoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
        NUMPY: _ClassVar[TrainingExample.Encoding]
        PYTORCH: _ClassVar[TrainingExample.Encoding]
    NUMPY: TrainingExample.Encoding
    PYTORCH: TrainingExample.Encoding
    class Stats(_message.Message):
        __slots__ = ["move", "duration_micros", "valid_moves", "visit_count"]
        MOVE_FIELD_NUMBER: _ClassVar[int]
        DURATION_MICROS_FIELD_NUMBER: _ClassVar[int]
        VALID_MOVES_FIELD_NUMBER: _ClassVar[int]
        VISIT_COUNT_FIELD_NUMBER: _ClassVar[int]
        move: int
        duration_micros: int
        valid_moves: int
        visit_count: int
        def __init__(self, move: _Optional[int] = ..., duration_micros: _Optional[int] = ..., valid_moves: _Optional[int] = ..., visit_count: _Optional[int] = ...) -> None: ...
    UNIX_MICROS_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    BOARD_FIELD_NUMBER: _ClassVar[int]
    MOVE_PROBS_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    unix_micros: int
    turn: int
    encoding: TrainingExample.Encoding
    board: bytes
    move_probs: bytes
    result: float
    stats: TrainingExample.Stats
    def __init__(self, unix_micros: _Optional[int] = ..., turn: _Optional[int] = ..., encoding: _Optional[_Union[TrainingExample.Encoding, str]] = ..., board: _Optional[bytes] = ..., move_probs: _Optional[bytes] = ..., result: _Optional[float] = ..., stats: _Optional[_Union[TrainingExample.Stats, _Mapping]] = ...) -> None: ...
