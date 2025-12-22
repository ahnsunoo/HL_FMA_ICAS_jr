cmake_minimum_required(VERSION 3.0.2)
project(tl_ctrl)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py.
## We don't use setup.py, so keep this commented.
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## (No custom messages/services/actions in this package)

# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# generate_messages(
#   DEPENDENCIES
#   geometry_msgs
#   sensor_msgs
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

# generate_dynamic_reconfigure_options(
#   # cfg/DynReconf1.cfg
# )

###################################
## catkin specific configuration ##
###################################
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES tl_ctrl
  CATKIN_DEPENDS cv_bridge geometry_msgs rospy sensor_msgs std_msgs
  # DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
)

## (No C++ targets for this package)
# add_library(${PROJECT_NAME} src/${PROJECT_NAME}/tl_ctrl.cpp)
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_executable(${PROJECT_NAME}_node src/tl_ctrl_node.cpp)
# target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})

#############
## Install ##
#############

## Install Python executables (scripts/)
## ※ 아래 목록의 파일들이 실제로 존재해야 catkin_make가 성공합니다.
catkin_install_python(
  PROGRAMS
    scripts/webcam.py
    scripts/l515_yolov5_traffic_light_node.py
    scripts/l515_yolov5_entry_stop_node.py
    # scripts/tl_obs_combo_node.py        # 만들었으면 주석 해제
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## (선택) launch 폴더 통째로 설치
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

# catkin_add_gtest(${PROJECT_NAME}-test test/test_tl_ctrl.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# catkin_add_nosetests(test)

