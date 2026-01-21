// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <atomic>
#include "index_context.h"
#include "index_helper.h"
#include "index_provider.h"
#include "index_runner.h"
#include "index_stats.h"

namespace zvec {
namespace core {

/*! Index Streamer
 */
class IndexStreamer : public IndexRunner {
 public:
  //! Index Streamer Pointer
  typedef std::shared_ptr<IndexStreamer> Pointer;

  //! Destructor
  virtual ~IndexStreamer(void) = default;

  //! Initialize the builder
  virtual int init(const IndexMeta & /*meta*/,
                   const ailego::Params & /*params*/) {
    return IndexError_NotImplemented;
  }

  //! Open a index from storage
  virtual int open(IndexStorage::Pointer stg) = 0;

  //! Flush index
  virtual int flush(uint64_t check_point) = 0;

  //! Close index
  virtual int close(void) = 0;

  //! Retrieve meta of index
  virtual const IndexMeta &meta(void) const = 0;
};

}  // namespace core
}  // namespace zvec
