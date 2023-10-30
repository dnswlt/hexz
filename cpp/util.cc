#include <chrono>
#include <string>

namespace hexz {

std::string GetEnv(const std::string& name) {
    const char* value = std::getenv(name.c_str());
    if (value == nullptr) {
        return "";
    }
    return std::string(value);
}

int GetEnvAsInt(const std::string& name, int default_value) {
    const char* value = std::getenv(name.c_str());
    if (value == nullptr) {
        return default_value;
    }
    return std::atoi(value);
}

int64_t UnixMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace hexz
