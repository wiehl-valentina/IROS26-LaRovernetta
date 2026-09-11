cmake_minimum_required(VERSION 3.8)
project(mini_plus_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_localization REQUIRED)
find_package(cv_bridge REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# --- INSTALACIÓN DE DIRECTORIOS COMPLEMENTARIOS ---
install(DIRECTORY 
  config 
  launch
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# --- NODOS PYTHON ---
install(PROGRAMS
  scripts/ekf_heading_bridge.py
  scripts/gps_course_heading_bridge.py
  scripts/udp_heading_exporter.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()