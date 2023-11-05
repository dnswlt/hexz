#ifndef __HEXZ_PYLIB_H__
#define __HEXZ_PYLIB_H__

#include <string>

namespace hexz {

// MoveSuggester is used as the interface to the C++ implementation from Python.
// It uses the pimpl idiom[1] to ensure that the Cython extension doesn't have
// to parse any libtorch and other heavyweight headers.
//
// [1] https://herbsutter.com/gotw/_100/
class MoveSuggester {
 public:
  MoveSuggester();
  // We need to declare the d'tor and define it in the .cc file
  // to make pimpl with a std::unique_ptr work.
  ~MoveSuggester();
  MoveSuggester(const MoveSuggester&) = delete;
  MoveSuggester(MoveSuggester&&);
  MoveSuggester& operator=(const MoveSuggester&) = delete;
  MoveSuggester& operator=(MoveSuggester&&);

  // Accepts a serialized hexzpb::SuggestMoveRequest and suggests a move.
  // Returns a serialized hexzpb::SuggestMoveResponse.
  // Raises std::invalid_argument if anything goes wrong.
  std::string SuggestMove(const std::string& request);

  // Loads the model at the given path.
  // Raises std::invalid_argument if there is no path at the given model
  // and std::runtime_exception if anything else goes wrong.
  void LoadModel(const std::string& path);

 private:
  class impl;
  impl* pimpl() { return pimpl_.get(); }
  const impl* pimpl() const { return pimpl_.get(); }
  std::unique_ptr<impl> pimpl_;
};

}  // namespace hexz

#endif  // __HEXZ_PYLIB_H__
