// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.5.1
// - protoc             v5.27.2
// source: hexzpb/hexz.proto

package hexzpb

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.64.0 or later.
const _ = grpc.SupportPackageIsVersion9

const (
	TrainingService_AddTrainingExamples_FullMethodName = "/hexzpb.TrainingService/AddTrainingExamples"
	TrainingService_FetchModel_FullMethodName          = "/hexzpb.TrainingService/FetchModel"
	TrainingService_ControlEvents_FullMethodName       = "/hexzpb.TrainingService/ControlEvents"
)

// TrainingServiceClient is the client API for TrainingService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type TrainingServiceClient interface {
	AddTrainingExamples(ctx context.Context, in *AddTrainingExamplesRequest, opts ...grpc.CallOption) (*AddTrainingExamplesResponse, error)
	FetchModel(ctx context.Context, in *FetchModelRequest, opts ...grpc.CallOption) (*FetchModelResponse, error)
	ControlEvents(ctx context.Context, in *ControlRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ControlEvent], error)
}

type trainingServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewTrainingServiceClient(cc grpc.ClientConnInterface) TrainingServiceClient {
	return &trainingServiceClient{cc}
}

func (c *trainingServiceClient) AddTrainingExamples(ctx context.Context, in *AddTrainingExamplesRequest, opts ...grpc.CallOption) (*AddTrainingExamplesResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(AddTrainingExamplesResponse)
	err := c.cc.Invoke(ctx, TrainingService_AddTrainingExamples_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *trainingServiceClient) FetchModel(ctx context.Context, in *FetchModelRequest, opts ...grpc.CallOption) (*FetchModelResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(FetchModelResponse)
	err := c.cc.Invoke(ctx, TrainingService_FetchModel_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *trainingServiceClient) ControlEvents(ctx context.Context, in *ControlRequest, opts ...grpc.CallOption) (grpc.ServerStreamingClient[ControlEvent], error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	stream, err := c.cc.NewStream(ctx, &TrainingService_ServiceDesc.Streams[0], TrainingService_ControlEvents_FullMethodName, cOpts...)
	if err != nil {
		return nil, err
	}
	x := &grpc.GenericClientStream[ControlRequest, ControlEvent]{ClientStream: stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type TrainingService_ControlEventsClient = grpc.ServerStreamingClient[ControlEvent]

// TrainingServiceServer is the server API for TrainingService service.
// All implementations must embed UnimplementedTrainingServiceServer
// for forward compatibility.
type TrainingServiceServer interface {
	AddTrainingExamples(context.Context, *AddTrainingExamplesRequest) (*AddTrainingExamplesResponse, error)
	FetchModel(context.Context, *FetchModelRequest) (*FetchModelResponse, error)
	ControlEvents(*ControlRequest, grpc.ServerStreamingServer[ControlEvent]) error
	mustEmbedUnimplementedTrainingServiceServer()
}

// UnimplementedTrainingServiceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedTrainingServiceServer struct{}

func (UnimplementedTrainingServiceServer) AddTrainingExamples(context.Context, *AddTrainingExamplesRequest) (*AddTrainingExamplesResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method AddTrainingExamples not implemented")
}
func (UnimplementedTrainingServiceServer) FetchModel(context.Context, *FetchModelRequest) (*FetchModelResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method FetchModel not implemented")
}
func (UnimplementedTrainingServiceServer) ControlEvents(*ControlRequest, grpc.ServerStreamingServer[ControlEvent]) error {
	return status.Errorf(codes.Unimplemented, "method ControlEvents not implemented")
}
func (UnimplementedTrainingServiceServer) mustEmbedUnimplementedTrainingServiceServer() {}
func (UnimplementedTrainingServiceServer) testEmbeddedByValue()                         {}

// UnsafeTrainingServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to TrainingServiceServer will
// result in compilation errors.
type UnsafeTrainingServiceServer interface {
	mustEmbedUnimplementedTrainingServiceServer()
}

func RegisterTrainingServiceServer(s grpc.ServiceRegistrar, srv TrainingServiceServer) {
	// If the following call pancis, it indicates UnimplementedTrainingServiceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&TrainingService_ServiceDesc, srv)
}

func _TrainingService_AddTrainingExamples_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(AddTrainingExamplesRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TrainingServiceServer).AddTrainingExamples(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: TrainingService_AddTrainingExamples_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TrainingServiceServer).AddTrainingExamples(ctx, req.(*AddTrainingExamplesRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _TrainingService_FetchModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(FetchModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TrainingServiceServer).FetchModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: TrainingService_FetchModel_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TrainingServiceServer).FetchModel(ctx, req.(*FetchModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _TrainingService_ControlEvents_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ControlRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(TrainingServiceServer).ControlEvents(m, &grpc.GenericServerStream[ControlRequest, ControlEvent]{ServerStream: stream})
}

// This type alias is provided for backwards compatibility with existing code that references the prior non-generic stream type by name.
type TrainingService_ControlEventsServer = grpc.ServerStreamingServer[ControlEvent]

// TrainingService_ServiceDesc is the grpc.ServiceDesc for TrainingService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var TrainingService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "hexzpb.TrainingService",
	HandlerType: (*TrainingServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "AddTrainingExamples",
			Handler:    _TrainingService_AddTrainingExamples_Handler,
		},
		{
			MethodName: "FetchModel",
			Handler:    _TrainingService_FetchModel_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "ControlEvents",
			Handler:       _TrainingService_ControlEvents_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "hexzpb/hexz.proto",
}

const (
	CPUPlayerService_SuggestMove_FullMethodName = "/hexzpb.CPUPlayerService/SuggestMove"
)

// CPUPlayerServiceClient is the client API for CPUPlayerService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type CPUPlayerServiceClient interface {
	SuggestMove(ctx context.Context, in *SuggestMoveRequest, opts ...grpc.CallOption) (*SuggestMoveResponse, error)
}

type cPUPlayerServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewCPUPlayerServiceClient(cc grpc.ClientConnInterface) CPUPlayerServiceClient {
	return &cPUPlayerServiceClient{cc}
}

func (c *cPUPlayerServiceClient) SuggestMove(ctx context.Context, in *SuggestMoveRequest, opts ...grpc.CallOption) (*SuggestMoveResponse, error) {
	cOpts := append([]grpc.CallOption{grpc.StaticMethod()}, opts...)
	out := new(SuggestMoveResponse)
	err := c.cc.Invoke(ctx, CPUPlayerService_SuggestMove_FullMethodName, in, out, cOpts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// CPUPlayerServiceServer is the server API for CPUPlayerService service.
// All implementations must embed UnimplementedCPUPlayerServiceServer
// for forward compatibility.
type CPUPlayerServiceServer interface {
	SuggestMove(context.Context, *SuggestMoveRequest) (*SuggestMoveResponse, error)
	mustEmbedUnimplementedCPUPlayerServiceServer()
}

// UnimplementedCPUPlayerServiceServer must be embedded to have
// forward compatible implementations.
//
// NOTE: this should be embedded by value instead of pointer to avoid a nil
// pointer dereference when methods are called.
type UnimplementedCPUPlayerServiceServer struct{}

func (UnimplementedCPUPlayerServiceServer) SuggestMove(context.Context, *SuggestMoveRequest) (*SuggestMoveResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SuggestMove not implemented")
}
func (UnimplementedCPUPlayerServiceServer) mustEmbedUnimplementedCPUPlayerServiceServer() {}
func (UnimplementedCPUPlayerServiceServer) testEmbeddedByValue()                          {}

// UnsafeCPUPlayerServiceServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to CPUPlayerServiceServer will
// result in compilation errors.
type UnsafeCPUPlayerServiceServer interface {
	mustEmbedUnimplementedCPUPlayerServiceServer()
}

func RegisterCPUPlayerServiceServer(s grpc.ServiceRegistrar, srv CPUPlayerServiceServer) {
	// If the following call pancis, it indicates UnimplementedCPUPlayerServiceServer was
	// embedded by pointer and is nil.  This will cause panics if an
	// unimplemented method is ever invoked, so we test this at initialization
	// time to prevent it from happening at runtime later due to I/O.
	if t, ok := srv.(interface{ testEmbeddedByValue() }); ok {
		t.testEmbeddedByValue()
	}
	s.RegisterService(&CPUPlayerService_ServiceDesc, srv)
}

func _CPUPlayerService_SuggestMove_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SuggestMoveRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(CPUPlayerServiceServer).SuggestMove(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: CPUPlayerService_SuggestMove_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(CPUPlayerServiceServer).SuggestMove(ctx, req.(*SuggestMoveRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// CPUPlayerService_ServiceDesc is the grpc.ServiceDesc for CPUPlayerService service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var CPUPlayerService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "hexzpb.CPUPlayerService",
	HandlerType: (*CPUPlayerServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SuggestMove",
			Handler:    _CPUPlayerService_SuggestMove_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "hexzpb/hexz.proto",
}