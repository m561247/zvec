cmake_minimum_required(VERSION 3.13)
cmake_policy(SET CMP0077 NEW)
project(zvec-example-c++)
set(CMAKE_CXX_STANDARD 17)

# Enable compile commands generation
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(../../src/include)
link_directories(../../build/lib ../../build/external/usr/local/lib)

add_executable(example main.cc)
target_link_libraries(
    example PRIVATE
    zvec_db
    zvec_ailego
    roaring
    rocksdb
    zvec_proto
    arrow
    arrow_acero
    arrow_bundled_dependencies
    arrow_compute
    arrow_dataset
    parquet
    ../../build/thirdparty/antlr/antlr4/runtime/Cpp/runtime/output/lib/libantlr4-runtime.a
    glog
    gflags_nothreads
    protobuf
    core_interface
    core_framework
    lz4
)
