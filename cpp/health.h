#pragma once

#include "health.grpc.pb.h"

namespace hexz {

class HealthServiceImpl final : public grpc::health::v1::Health::Service {
 public:
  grpc::Status Check(grpc::ServerContext* context,
                     const grpc::health::v1::HealthCheckRequest* request,
                     grpc::health::v1::HealthCheckResponse* response) override;
  grpc::Status Watch(
      grpc::ServerContext* context,
      const grpc::health::v1::HealthCheckRequest* request,
      grpc::ServerWriter<grpc::health::v1::HealthCheckResponse>* writer) override;
};

}  // namespace hexz
