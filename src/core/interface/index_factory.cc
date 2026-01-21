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
//
// Created by wangjianning.wjn on 8/28/25.
//

#include <zvec/core/framework/index_factory.h>
#include "ailego/container/params.h"
#include <zvec/core/framework/index_meta.h>
#include <zvec/core/interface/index_factory.h>
#include <zvec/core/interface/utils/utils.h>

namespace zvec::core_interface {


Index::Pointer IndexFactory::CreateAndInitIndex(const BaseIndexParam &param) {
  Index::Pointer ptr = nullptr;
  // if (param.index_type == IndexType::kIVF) {
  //   const IVFIndexParam *_param = dynamic_cast<const IVFIndexParam
  //   *>(&param); ptr = std::make_shared<IVFIndex>(param);

  //   if (_param->l1Index) {
  //     // TODO: create l1 index
  //   }
  //   if (_param->l2Index) {
  //     // TODO: create l2 index
  //   }
  // }
  // if (param.index_type == IndexType::kHNSW) {
  //   ptr = std::make_shared<HNSWIndex>(param);
  // }
  if (param.index_type == IndexType::kFlat) {
    // ptr = std::make_shared<FlatIndex>(param);
    ptr = std::make_shared<FlatIndex>();
  } else if (param.index_type == IndexType::kHNSW) {
    ptr = std::make_shared<HNSWIndex>();
  } else if (param.index_type == IndexType::kIVF) {
    ptr = std::make_shared<IVFIndex>();
  } else {
    LOG_ERROR("Unsupported index type: ");
    return nullptr;
  }

  if (!ptr) {
    LOG_ERROR("Failed to create index");
    return nullptr;
  }
  if (0 != ptr->Init(param)) {
    LOG_ERROR("Failed to init index");
    return nullptr;
  }
  return ptr;
}

BaseIndexParam::Pointer IndexFactory::DeserializeIndexParamFromJson(
    const std::string &json_str) {
  ailego::JsonValue json_value;
  if (!json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = json_value.as_object();
  ailego::JsonValue tmp_json_value;

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  switch (index_type) {
    case IndexType::kFlat: {
      FlatIndexParam::Pointer param = std::make_shared<FlatIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize flat index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kHNSW: {
      HNSWIndexParam::Pointer param = std::make_shared<HNSWIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    case IndexType::kIVF: {
      IVFIndexParam::Pointer param = std::make_shared<IVFIndexParam>();
      if (!param->DeserializeFromJson(json_str)) {
        LOG_ERROR("Failed to deserialize hnsw index param");
        return nullptr;
      }
      return param;
    }
    default:
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
  }
}

}  // namespace zvec::core_interface
