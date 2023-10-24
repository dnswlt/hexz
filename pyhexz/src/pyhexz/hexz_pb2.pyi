from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar, Iterable, Mapping, Optional, Union

DESCRIPTOR: _descriptor.FileDescriptor

class AddTrainingExamplesRequest(_message.Message):
    __slots__ = ["examples", "model_key"]
    EXAMPLES_FIELD_NUMBER: ClassVar[int]
    MODEL_KEY_FIELD_NUMBER: ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[TrainingExample]
    model_key: ModelKey
    def __init__(self, model_key: Optional[Union[ModelKey, Mapping]] = ..., examples: Optional[Iterable[Union[TrainingExample, Mapping]]] = ...) -> None: ...

class AddTrainingExamplesResponse(_message.Message):
    __slots__ = ["error_message", "latest_model", "status"]
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ACCEPTED: AddTrainingExamplesResponse.Status
    ERROR_MESSAGE_FIELD_NUMBER: ClassVar[int]
    LATEST_MODEL_FIELD_NUMBER: ClassVar[int]
    REJECTED: AddTrainingExamplesResponse.Status
    STATUS_FIELD_NUMBER: ClassVar[int]
    STATUS_UNSPECIFIED: AddTrainingExamplesResponse.Status
    error_message: str
    latest_model: ModelKey
    status: AddTrainingExamplesResponse.Status
    def __init__(self, status: Optional[Union[AddTrainingExamplesResponse.Status, str]] = ..., latest_model: Optional[Union[ModelKey, Mapping]] = ..., error_message: Optional[str] = ...) -> None: ...

class Board(_message.Message):
    __slots__ = ["flat_fields", "last_revealed", "move", "resources", "score", "state", "turn"]
    class GameState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    FINISHED: Board.GameState
    FLAT_FIELDS_FIELD_NUMBER: ClassVar[int]
    INITIAL: Board.GameState
    LAST_REVEALED_FIELD_NUMBER: ClassVar[int]
    MOVE_FIELD_NUMBER: ClassVar[int]
    RESOURCES_FIELD_NUMBER: ClassVar[int]
    RUNNING: Board.GameState
    SCORE_FIELD_NUMBER: ClassVar[int]
    STATE_FIELD_NUMBER: ClassVar[int]
    TURN_FIELD_NUMBER: ClassVar[int]
    flat_fields: _containers.RepeatedCompositeFieldContainer[Field]
    last_revealed: int
    move: int
    resources: _containers.RepeatedCompositeFieldContainer[ResourceInfo]
    score: _containers.RepeatedScalarFieldContainer[int]
    state: Board.GameState
    turn: int
    def __init__(self, turn: Optional[int] = ..., move: Optional[int] = ..., last_revealed: Optional[int] = ..., flat_fields: Optional[Iterable[Union[Field, Mapping]]] = ..., score: Optional[Iterable[int]] = ..., resources: Optional[Iterable[Union[ResourceInfo, Mapping]]] = ..., state: Optional[Union[Board.GameState, str]] = ...) -> None: ...

class Field(_message.Message):
    __slots__ = ["blocked", "hidden", "lifetime", "next_val", "owner", "type", "value"]
    class CellType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BLOCKED_FIELD_NUMBER: ClassVar[int]
    DEAD: Field.CellType
    DEATH: Field.CellType
    FIRE: Field.CellType
    FLAG: Field.CellType
    GRASS: Field.CellType
    HIDDEN_FIELD_NUMBER: ClassVar[int]
    LIFETIME_FIELD_NUMBER: ClassVar[int]
    NEXT_VAL_FIELD_NUMBER: ClassVar[int]
    NORMAL: Field.CellType
    OWNER_FIELD_NUMBER: ClassVar[int]
    PEST: Field.CellType
    ROCK: Field.CellType
    TYPE_FIELD_NUMBER: ClassVar[int]
    VALUE_FIELD_NUMBER: ClassVar[int]
    blocked: int
    hidden: bool
    lifetime: int
    next_val: _containers.RepeatedScalarFieldContainer[int]
    owner: int
    type: Field.CellType
    value: int
    def __init__(self, type: Optional[Union[Field.CellType, str]] = ..., owner: Optional[int] = ..., hidden: bool = ..., value: Optional[int] = ..., blocked: Optional[int] = ..., lifetime: Optional[int] = ..., next_val: Optional[Iterable[int]] = ...) -> None: ...

class GameEngineClassicState(_message.Message):
    __slots__ = ["board"]
    BOARD_FIELD_NUMBER: ClassVar[int]
    board: Board
    def __init__(self, board: Optional[Union[Board, Mapping]] = ...) -> None: ...

class GameEngineFlagzState(_message.Message):
    __slots__ = ["board", "free_cells", "moves", "normal_moves"]
    BOARD_FIELD_NUMBER: ClassVar[int]
    FREE_CELLS_FIELD_NUMBER: ClassVar[int]
    MOVES_FIELD_NUMBER: ClassVar[int]
    NORMAL_MOVES_FIELD_NUMBER: ClassVar[int]
    board: Board
    free_cells: int
    moves: _containers.RepeatedCompositeFieldContainer[GameEngineMove]
    normal_moves: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, board: Optional[Union[Board, Mapping]] = ..., free_cells: Optional[int] = ..., normal_moves: Optional[Iterable[int]] = ..., moves: Optional[Iterable[Union[GameEngineMove, Mapping]]] = ...) -> None: ...

class GameEngineFreeformState(_message.Message):
    __slots__ = ["board"]
    BOARD_FIELD_NUMBER: ClassVar[int]
    board: Board
    def __init__(self, board: Optional[Union[Board, Mapping]] = ...) -> None: ...

class GameEngineMove(_message.Message):
    __slots__ = ["cell_type", "col", "move", "player_num", "row"]
    CELL_TYPE_FIELD_NUMBER: ClassVar[int]
    COL_FIELD_NUMBER: ClassVar[int]
    MOVE_FIELD_NUMBER: ClassVar[int]
    PLAYER_NUM_FIELD_NUMBER: ClassVar[int]
    ROW_FIELD_NUMBER: ClassVar[int]
    cell_type: Field.CellType
    col: int
    move: int
    player_num: int
    row: int
    def __init__(self, player_num: Optional[int] = ..., move: Optional[int] = ..., row: Optional[int] = ..., col: Optional[int] = ..., cell_type: Optional[Union[Field.CellType, str]] = ...) -> None: ...

class GameEngineState(_message.Message):
    __slots__ = ["classic", "flagz", "freeform"]
    CLASSIC_FIELD_NUMBER: ClassVar[int]
    FLAGZ_FIELD_NUMBER: ClassVar[int]
    FREEFORM_FIELD_NUMBER: ClassVar[int]
    classic: GameEngineClassicState
    flagz: GameEngineFlagzState
    freeform: GameEngineFreeformState
    def __init__(self, flagz: Optional[Union[GameEngineFlagzState, Mapping]] = ..., classic: Optional[Union[GameEngineClassicState, Mapping]] = ..., freeform: Optional[Union[GameEngineFreeformState, Mapping]] = ...) -> None: ...

class GameInfo(_message.Message):
    __slots__ = ["cpu_player", "host", "id", "started", "type"]
    CPU_PLAYER_FIELD_NUMBER: ClassVar[int]
    HOST_FIELD_NUMBER: ClassVar[int]
    ID_FIELD_NUMBER: ClassVar[int]
    STARTED_FIELD_NUMBER: ClassVar[int]
    TYPE_FIELD_NUMBER: ClassVar[int]
    cpu_player: bool
    host: str
    id: str
    started: _timestamp_pb2.Timestamp
    type: str
    def __init__(self, id: Optional[str] = ..., host: Optional[str] = ..., started: Optional[Union[_timestamp_pb2.Timestamp, Mapping]] = ..., type: Optional[str] = ..., cpu_player: bool = ...) -> None: ...

class GameState(_message.Message):
    __slots__ = ["engine_state", "game_info", "modified", "players", "seqnum"]
    ENGINE_STATE_FIELD_NUMBER: ClassVar[int]
    GAME_INFO_FIELD_NUMBER: ClassVar[int]
    MODIFIED_FIELD_NUMBER: ClassVar[int]
    PLAYERS_FIELD_NUMBER: ClassVar[int]
    SEQNUM_FIELD_NUMBER: ClassVar[int]
    engine_state: GameEngineState
    game_info: GameInfo
    modified: _timestamp_pb2.Timestamp
    players: _containers.RepeatedCompositeFieldContainer[Player]
    seqnum: int
    def __init__(self, game_info: Optional[Union[GameInfo, Mapping]] = ..., seqnum: Optional[int] = ..., modified: Optional[Union[_timestamp_pb2.Timestamp, Mapping]] = ..., players: Optional[Iterable[Union[Player, Mapping]]] = ..., engine_state: Optional[Union[GameEngineState, Mapping]] = ...) -> None: ...

class MCTSExample(_message.Message):
    __slots__ = ["board", "game_id", "move_stats", "result"]
    class MoveStats(_message.Message):
        __slots__ = ["move", "visits", "win_rate"]
        MOVE_FIELD_NUMBER: ClassVar[int]
        VISITS_FIELD_NUMBER: ClassVar[int]
        WIN_RATE_FIELD_NUMBER: ClassVar[int]
        move: GameEngineMove
        visits: int
        win_rate: float
        def __init__(self, move: Optional[Union[GameEngineMove, Mapping]] = ..., visits: Optional[int] = ..., win_rate: Optional[float] = ...) -> None: ...
    BOARD_FIELD_NUMBER: ClassVar[int]
    GAME_ID_FIELD_NUMBER: ClassVar[int]
    MOVE_STATS_FIELD_NUMBER: ClassVar[int]
    RESULT_FIELD_NUMBER: ClassVar[int]
    board: Board
    game_id: str
    move_stats: _containers.RepeatedCompositeFieldContainer[MCTSExample.MoveStats]
    result: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, game_id: Optional[str] = ..., board: Optional[Union[Board, Mapping]] = ..., result: Optional[Iterable[int]] = ..., move_stats: Optional[Iterable[Union[MCTSExample.MoveStats, Mapping]]] = ...) -> None: ...

class ModelKey(_message.Message):
    __slots__ = ["checkpoint", "name"]
    CHECKPOINT_FIELD_NUMBER: ClassVar[int]
    NAME_FIELD_NUMBER: ClassVar[int]
    checkpoint: int
    name: str
    def __init__(self, name: Optional[str] = ..., checkpoint: Optional[int] = ...) -> None: ...

class Player(_message.Message):
    __slots__ = ["id", "name"]
    ID_FIELD_NUMBER: ClassVar[int]
    NAME_FIELD_NUMBER: ClassVar[int]
    id: str
    name: str
    def __init__(self, id: Optional[str] = ..., name: Optional[str] = ...) -> None: ...

class ResourceInfo(_message.Message):
    __slots__ = ["num_pieces"]
    NUM_PIECES_FIELD_NUMBER: ClassVar[int]
    num_pieces: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, num_pieces: Optional[Iterable[int]] = ...) -> None: ...

class TrainingExample(_message.Message):
    __slots__ = ["board", "move_probs", "result", "unix_micros"]
    BOARD_FIELD_NUMBER: ClassVar[int]
    MOVE_PROBS_FIELD_NUMBER: ClassVar[int]
    RESULT_FIELD_NUMBER: ClassVar[int]
    UNIX_MICROS_FIELD_NUMBER: ClassVar[int]
    board: bytes
    move_probs: bytes
    result: float
    unix_micros: int
    def __init__(self, unix_micros: Optional[int] = ..., board: Optional[bytes] = ..., move_probs: Optional[bytes] = ..., result: Optional[float] = ...) -> None: ...
