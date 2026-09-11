cmake_minimum_required(VERSION 3.5)
project(er_bringup)

# Find the ament build system (CMake based)
find_package(ament_cmake REQUIRED)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
