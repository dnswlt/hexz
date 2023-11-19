# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: hexz.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\nhexz.proto\x12\x06hexzpb\x1a\x1fgoogle/protobuf/timestamp.proto\"\xf3\x01\n\x05\x42oard\x12\x0c\n\x04turn\x18\x01 \x01(\x05\x12\x0c\n\x04move\x18\x02 \x01(\x05\x12\x15\n\rlast_revealed\x18\x03 \x01(\x05\x12\"\n\x0b\x66lat_fields\x18\x04 \x03(\x0b\x32\r.hexzpb.Field\x12\r\n\x05score\x18\x05 \x03(\x05\x12\'\n\tresources\x18\x06 \x03(\x0b\x32\x14.hexzpb.ResourceInfo\x12&\n\x05state\x18\x07 \x01(\x0e\x32\x17.hexzpb.Board.GameState\"3\n\tGameState\x12\x0b\n\x07INITIAL\x10\x00\x12\x0b\n\x07RUNNING\x10\x01\x12\x0c\n\x08\x46INISHED\x10\x02\"\xf0\x01\n\x05\x46ield\x12$\n\x04type\x18\x01 \x01(\x0e\x32\x16.hexzpb.Field.CellType\x12\r\n\x05owner\x18\x02 \x01(\x05\x12\x0e\n\x06hidden\x18\x03 \x01(\x08\x12\r\n\x05value\x18\x04 \x01(\x05\x12\x0f\n\x07\x62locked\x18\x05 \x01(\x05\x12\x10\n\x08lifetime\x18\x06 \x01(\x05\x12\x10\n\x08next_val\x18\x07 \x03(\x05\"^\n\x08\x43\x65llType\x12\n\n\x06NORMAL\x10\x00\x12\x08\n\x04\x44\x45\x41\x44\x10\x01\x12\t\n\x05GRASS\x10\x02\x12\x08\n\x04ROCK\x10\x03\x12\x08\n\x04\x46IRE\x10\x04\x12\x08\n\x04\x46LAG\x10\x05\x12\x08\n\x04PEST\x10\x06\x12\t\n\x05\x44\x45\x41TH\x10\x07\"\"\n\x0cResourceInfo\x12\x12\n\nnum_pieces\x18\x01 \x03(\x05\"\"\n\x06Player\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\"s\n\x08GameInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04host\x18\x02 \x01(\t\x12+\n\x07started\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x12\n\ncpu_player\x18\x05 \x01(\x08\"\xbe\x01\n\tGameState\x12#\n\tgame_info\x18\x01 \x01(\x0b\x32\x10.hexzpb.GameInfo\x12\x0e\n\x06seqnum\x18\x02 \x01(\x03\x12,\n\x08modified\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x1f\n\x07players\x18\x05 \x03(\x0b\x32\x0e.hexzpb.Player\x12-\n\x0c\x65ngine_state\x18\x06 \x01(\x0b\x32\x17.hexzpb.GameEngineState\"\xb1\x01\n\x0fGameEngineState\x12-\n\x05\x66lagz\x18\x01 \x01(\x0b\x32\x1c.hexzpb.GameEngineFlagzStateH\x00\x12\x31\n\x07\x63lassic\x18\x02 \x01(\x0b\x32\x1e.hexzpb.GameEngineClassicStateH\x00\x12\x33\n\x08\x66reeform\x18\x03 \x01(\x0b\x32\x1f.hexzpb.GameEngineFreeformStateH\x00\x42\x07\n\x05state\"\x85\x01\n\x14GameEngineFlagzState\x12\x1c\n\x05\x62oard\x18\x01 \x01(\x0b\x32\r.hexzpb.Board\x12\x12\n\nfree_cells\x18\x02 \x01(\x05\x12\x14\n\x0cnormal_moves\x18\x03 \x03(\x05\x12%\n\x05moves\x18\x04 \x03(\x0b\x32\x16.hexzpb.GameEngineMove\"6\n\x16GameEngineClassicState\x12\x1c\n\x05\x62oard\x18\x01 \x01(\x0b\x32\r.hexzpb.Board\"7\n\x17GameEngineFreeformState\x12\x1c\n\x05\x62oard\x18\x01 \x01(\x0b\x32\r.hexzpb.Board\"w\n\x0eGameEngineMove\x12\x12\n\nplayer_num\x18\x01 \x01(\x05\x12\x0c\n\x04move\x18\x02 \x01(\x05\x12\x0b\n\x03row\x18\x03 \x01(\x05\x12\x0b\n\x03\x63ol\x18\x04 \x01(\x05\x12)\n\tcell_type\x18\x05 \x01(\x0e\x32\x16.hexzpb.Field.CellType\"\xd4\x01\n\x0bMCTSExample\x12\x0f\n\x07game_id\x18\x01 \x01(\t\x12\x1c\n\x05\x62oard\x18\x02 \x01(\x0b\x32\r.hexzpb.Board\x12\x0e\n\x06result\x18\x03 \x03(\x05\x12\x31\n\nmove_stats\x18\x04 \x03(\x0b\x32\x1d.hexzpb.MCTSExample.MoveStats\x1aS\n\tMoveStats\x12$\n\x04move\x18\x01 \x01(\x0b\x32\x16.hexzpb.GameEngineMove\x12\x0e\n\x06visits\x18\x02 \x01(\x05\x12\x10\n\x08win_rate\x18\x03 \x01(\x02\"c\n\x12SuggestMoveRequest\x12\x19\n\x11max_think_time_ms\x18\x01 \x01(\x03\x12\x32\n\x11game_engine_state\x18\x02 \x01(\x0b\x32\x17.hexzpb.GameEngineState\"\xc5\x02\n\x10SuggestMoveStats\x12\x32\n\x05moves\x18\x01 \x03(\x0b\x32#.hexzpb.SuggestMoveStats.ScoredMove\x12\r\n\x05value\x18\x02 \x01(\x02\x1aH\n\x05Score\x12\x30\n\x04kind\x18\x01 \x01(\x0e\x32\".hexzpb.SuggestMoveStats.ScoreKind\x12\r\n\x05score\x18\x02 \x01(\x02\x1a|\n\nScoredMove\x12\x0b\n\x03row\x18\x01 \x01(\x05\x12\x0b\n\x03\x63ol\x18\x02 \x01(\x05\x12$\n\x04type\x18\x03 \x01(\x0e\x32\x16.hexzpb.Field.CellType\x12.\n\x06scores\x18\x04 \x03(\x0b\x32\x1e.hexzpb.SuggestMoveStats.Score\"&\n\tScoreKind\x12\t\n\x05\x46INAL\x10\x00\x12\x0e\n\nMCTS_PRIOR\x10\x01\"i\n\x13SuggestMoveResponse\x12$\n\x04move\x18\x01 \x01(\x0b\x32\x16.hexzpb.GameEngineMove\x12,\n\nmove_stats\x18\x02 \x01(\x0b\x32\x18.hexzpb.SuggestMoveStats\",\n\x08ModelKey\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ncheckpoint\x18\x02 \x01(\x05\"l\n\x1a\x41\x64\x64TrainingExamplesRequest\x12#\n\tmodel_key\x18\x01 \x01(\x0b\x32\x10.hexzpb.ModelKey\x12)\n\x08\x65xamples\x18\x02 \x03(\x0b\x32\x17.hexzpb.TrainingExample\"\x90\x02\n\x1b\x41\x64\x64TrainingExamplesResponse\x12:\n\x06status\x18\x01 \x01(\x0e\x32*.hexzpb.AddTrainingExamplesResponse.Status\x12&\n\x0clatest_model\x18\x02 \x01(\x0b\x32\x10.hexzpb.ModelKey\x12\x15\n\rerror_message\x18\x03 \x01(\t\"v\n\x06Status\x12\x16\n\x12STATUS_UNSPECIFIED\x10\x00\x12\x0c\n\x08\x41\x43\x43\x45PTED\x10\x01\x12\x18\n\x14REJECTED_WRONG_MODEL\x10\x02\x12\x18\n\x14REJECTED_AT_CAPACITY\x10\x03\x12\x12\n\x0eREJECTED_OTHER\x10\x04\"\xf4\x02\n\x0fTrainingExample\x12\x13\n\x0bunix_micros\x18\x01 \x01(\x03\x12\x0c\n\x04turn\x18\x07 \x01(\x05\x12$\n\x04move\x18\t \x01(\x0b\x32\x16.hexzpb.GameEngineMove\x12\x32\n\x08\x65ncoding\x18\x06 \x01(\x0e\x32 .hexzpb.TrainingExample.Encoding\x12\r\n\x05\x62oard\x18\x02 \x01(\x0c\x12\x13\n\x0b\x61\x63tion_mask\x18\x08 \x01(\x0c\x12\x12\n\nmove_probs\x18\x03 \x01(\x0c\x12\x0e\n\x06result\x18\x04 \x01(\x02\x12,\n\x05stats\x18\x05 \x01(\x0b\x32\x1d.hexzpb.TrainingExample.Stats\x1aJ\n\x05Stats\x12\x17\n\x0f\x64uration_micros\x18\x03 \x01(\x03\x12\x13\n\x0bvalid_moves\x18\x04 \x01(\x05\x12\x13\n\x0bvisit_count\x18\x05 \x01(\x05\"\"\n\x08\x45ncoding\x12\t\n\x05NUMPY\x10\x00\x12\x0b\n\x07PYTORCH\x10\x01\x42\x1fZ\x1dgithub.com/dnswlt/hexz/hexzpbb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'hexz_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'Z\035github.com/dnswlt/hexz/hexzpb'
  _globals['_BOARD']._serialized_start=56
  _globals['_BOARD']._serialized_end=299
  _globals['_BOARD_GAMESTATE']._serialized_start=248
  _globals['_BOARD_GAMESTATE']._serialized_end=299
  _globals['_FIELD']._serialized_start=302
  _globals['_FIELD']._serialized_end=542
  _globals['_FIELD_CELLTYPE']._serialized_start=448
  _globals['_FIELD_CELLTYPE']._serialized_end=542
  _globals['_RESOURCEINFO']._serialized_start=544
  _globals['_RESOURCEINFO']._serialized_end=578
  _globals['_PLAYER']._serialized_start=580
  _globals['_PLAYER']._serialized_end=614
  _globals['_GAMEINFO']._serialized_start=616
  _globals['_GAMEINFO']._serialized_end=731
  _globals['_GAMESTATE']._serialized_start=734
  _globals['_GAMESTATE']._serialized_end=924
  _globals['_GAMEENGINESTATE']._serialized_start=927
  _globals['_GAMEENGINESTATE']._serialized_end=1104
  _globals['_GAMEENGINEFLAGZSTATE']._serialized_start=1107
  _globals['_GAMEENGINEFLAGZSTATE']._serialized_end=1240
  _globals['_GAMEENGINECLASSICSTATE']._serialized_start=1242
  _globals['_GAMEENGINECLASSICSTATE']._serialized_end=1296
  _globals['_GAMEENGINEFREEFORMSTATE']._serialized_start=1298
  _globals['_GAMEENGINEFREEFORMSTATE']._serialized_end=1353
  _globals['_GAMEENGINEMOVE']._serialized_start=1355
  _globals['_GAMEENGINEMOVE']._serialized_end=1474
  _globals['_MCTSEXAMPLE']._serialized_start=1477
  _globals['_MCTSEXAMPLE']._serialized_end=1689
  _globals['_MCTSEXAMPLE_MOVESTATS']._serialized_start=1606
  _globals['_MCTSEXAMPLE_MOVESTATS']._serialized_end=1689
  _globals['_SUGGESTMOVEREQUEST']._serialized_start=1691
  _globals['_SUGGESTMOVEREQUEST']._serialized_end=1790
  _globals['_SUGGESTMOVESTATS']._serialized_start=1793
  _globals['_SUGGESTMOVESTATS']._serialized_end=2118
  _globals['_SUGGESTMOVESTATS_SCORE']._serialized_start=1880
  _globals['_SUGGESTMOVESTATS_SCORE']._serialized_end=1952
  _globals['_SUGGESTMOVESTATS_SCOREDMOVE']._serialized_start=1954
  _globals['_SUGGESTMOVESTATS_SCOREDMOVE']._serialized_end=2078
  _globals['_SUGGESTMOVESTATS_SCOREKIND']._serialized_start=2080
  _globals['_SUGGESTMOVESTATS_SCOREKIND']._serialized_end=2118
  _globals['_SUGGESTMOVERESPONSE']._serialized_start=2120
  _globals['_SUGGESTMOVERESPONSE']._serialized_end=2225
  _globals['_MODELKEY']._serialized_start=2227
  _globals['_MODELKEY']._serialized_end=2271
  _globals['_ADDTRAININGEXAMPLESREQUEST']._serialized_start=2273
  _globals['_ADDTRAININGEXAMPLESREQUEST']._serialized_end=2381
  _globals['_ADDTRAININGEXAMPLESRESPONSE']._serialized_start=2384
  _globals['_ADDTRAININGEXAMPLESRESPONSE']._serialized_end=2656
  _globals['_ADDTRAININGEXAMPLESRESPONSE_STATUS']._serialized_start=2538
  _globals['_ADDTRAININGEXAMPLESRESPONSE_STATUS']._serialized_end=2656
  _globals['_TRAININGEXAMPLE']._serialized_start=2659
  _globals['_TRAININGEXAMPLE']._serialized_end=3031
  _globals['_TRAININGEXAMPLE_STATS']._serialized_start=2921
  _globals['_TRAININGEXAMPLE_STATS']._serialized_end=2995
  _globals['_TRAININGEXAMPLE_ENCODING']._serialized_start=2997
  _globals['_TRAININGEXAMPLE_ENCODING']._serialized_end=3031
# @@protoc_insertion_point(module_scope)
