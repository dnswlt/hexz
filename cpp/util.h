#ifndef __HEXZ_UTIL_H__
#define __HEXZ_UTIL_H__
#include <string>

namespace hexz {

std::string GetEnv(const std::string& name);
int GetEnvAsInt(const std::string& name, int default_value);

int64_t UnixMicros();

}  // namespace hexz
#endif  // __HEXZ_UTIL_H__