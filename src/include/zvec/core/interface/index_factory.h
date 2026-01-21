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

#ifndef ZVEC_INDEX_FACTORY_H
#define ZVEC_INDEX_FACTORY_H

#pragma once
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "index.h"
#include "index_param.h"

namespace zvec::core_interface {

// 索引的工厂类
class IndexFactory {
 public:
  static Index::Pointer CreateAndInitIndex(const BaseIndexParam &param);

  static BaseIndexParam::Pointer DeserializeIndexParamFromJson(
      const std::string &json_str);


  static std::string QueryParamSerializeToJson(
      const BaseIndexQueryParam &param);


  template <
      typename QueryParamType,
      std::enable_if_t<std::is_base_of_v<BaseIndexQueryParam, QueryParamType>,
                       bool> = true>
  static std::string QueryParamSerializeToJson(const QueryParamType &param,
                                               bool omit_empty_value = false);

  template <
      typename QueryParamType,
      std::enable_if_t<std::is_base_of_v<BaseIndexQueryParam, QueryParamType>,
                       bool> = true>
  static typename QueryParamType::Pointer QueryParamDeserializeFromJson(
      const std::string &json_str);

  // register() -- Index class should have a `create` interface
};


template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
std::string IndexFactory::QueryParamSerializeToJson(const QueryParamType &param,
                                                    bool omit_empty_value) {
  ailego::JsonObject json_obj;

  // BaseIndexQueryParam
  // omit filter & bf_pks
  if (!omit_empty_value || param.topk != 0) {
    json_obj.set("topk", ailego::JsonValue(param.topk));
  }
  if (!omit_empty_value || param.fetch_vector) {
    json_obj.set("fetch_vector", ailego::JsonValue(param.fetch_vector));
  }
  if (!omit_empty_value || param.radius != 0.0f) {
    json_obj.set("radius", ailego::JsonValue(param.radius));
  }
  if (!omit_empty_value || param.is_linear) {
    json_obj.set("is_linear", ailego::JsonValue(param.is_linear));
  }

  IndexType index_type;
  if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    // index_type
    index_type = IndexType::kFlat;
  } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
    if (!omit_empty_value || param.ef_search != 0) {
      json_obj.set("ef_search", ailego::JsonValue(param.ef_search));
    }
    index_type = IndexType::kHNSW;
  } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
    if (!omit_empty_value || param.nprobe != 0) {
      json_obj.set("nprobe", ailego::JsonValue(param.nprobe));
    }
    index_type = IndexType::kIVF;
    // json_obj.set("l1QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l1QueryParam)));
    // json_obj.set("l2QueryParam",
    // ailego::JsonValue(QueryParamSerializeToJson(param.l2QueryParam)));
  }

  json_obj.set("index_type",
               ailego::JsonValue(magic_enum::enum_name(index_type).data()));

  return ailego::JsonValue(json_obj).as_json_string().as_stl_string();
}

template <typename QueryParamType,
          std::enable_if_t<
              std::is_base_of_v<BaseIndexQueryParam, QueryParamType>, bool> >
typename QueryParamType::Pointer IndexFactory::QueryParamDeserializeFromJson(
    const std::string &json_str) {
  ailego::JsonValue tmp_json_value;
  if (!tmp_json_value.parse(json_str)) {
    LOG_ERROR("Failed to parse json string: %s", json_str.c_str());
    return nullptr;
  }
  ailego::JsonObject json_obj = tmp_json_value.as_object();

  auto parse_common_fields = [&](auto &param) -> bool {
    if (!extract_value_from_json(json_obj, "topk", param->topk,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize topk");
      return false;
    }

    if (!extract_value_from_json(json_obj, "fetch_vector", param->fetch_vector,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize fetch_vector");
      return false;
    }

    if (!extract_value_from_json(json_obj, "radius", param->radius,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize radius");
      return false;
    }

    if (!extract_value_from_json(json_obj, "is_linear", param->is_linear,
                                 tmp_json_value)) {
      LOG_ERROR("Failed to deserialize is_linear");
      return false;
    }
    return true;
  };

  IndexType index_type;

  if (!extract_enum_from_json<IndexType>(json_obj, "index_type", index_type,
                                         tmp_json_value)) {
    LOG_ERROR("Failed to deserialize index type");
    return nullptr;
  }

  if constexpr (std::is_same_v<QueryParamType, BaseIndexQueryParam>) {
    if (index_type == IndexType::kFlat) {
      auto param = std::make_shared<FlatQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kHNSW) {
      auto param = std::make_shared<HNSWQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
      return param;
    } else if (index_type == IndexType::kIVF) {
      auto param = std::make_shared<IVFQueryParam>();
      if (!parse_common_fields(param)) {
        return nullptr;
      }
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
      return param;
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
  } else {
    auto param = std::make_shared<QueryParamType>();
    if (!parse_common_fields(param)) {
      return nullptr;
    }
    if constexpr (std::is_same_v<QueryParamType, FlatQueryParam>) {
    } else if constexpr (std::is_same_v<QueryParamType, HNSWQueryParam>) {
      if (!extract_value_from_json(json_obj, "ef_search", param->ef_search,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize ef_search");
        return nullptr;
      }
    } else if constexpr (std::is_same_v<QueryParamType, IVFQueryParam>) {
      if (!extract_value_from_json(json_obj, "nprobe", param->nprobe,
                                   tmp_json_value)) {
        LOG_ERROR("Failed to deserialize nprobe");
        return nullptr;
      }
    } else {
      LOG_ERROR("Unsupported index type: %s",
                magic_enum::enum_name(index_type).data());
      return nullptr;
    }
    return param;
  }
}


}  // namespace zvec::core_interface


#endif  // ZVEC_INDEX_FACTORY_H
