#include "health.h"

#include <absl/log/absl_log.h>

#include <thread>

namespace hexz {

using grpc::health::v1::HealthCheckRequest;
using grpc::health::v1::HealthCheckResponse;

grpc::Status HealthServiceImpl::Check(grpc::ServerContext*,
                                      const HealthCheckRequest* request,
                                      HealthCheckResponse* response) {
  ABSL_LOG(INFO) << "Received health request for service "
                 << request->service();
  response->set_status(HealthCheckResponse::SERVING);
  return grpc::Status::OK;
}

grpc::Status HealthServiceImpl::Watch(
    grpc::ServerContext* context, const HealthCheckRequest* request,
    grpc::ServerWriter<HealthCheckResponse>* writer) {
  ABSL_LOG(INFO) << "Received health request for service '"
                 << request->service() << "'";

  HealthCheckResponse response;
  response.set_status(HealthCheckResponse::SERVING);

  while (!context->IsCancelled()) {
    writer->Write(response);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return grpc::Status::OK;
}

}  // namespace hexz
