Henes Autonomous Car Control System
이 저장소는 Henes 전동차를 이용한 자율주행 제어 시스템의 실행 프로세스를 담고 있습니다. GPS, IMU, 카메라 센서 데이터를 융합하여 주행 제어 및 장애물/신호등 인식을 수행합니다.

🛠 사전 준비 (Environment Setup)
터미널을 열고 하드웨어 연결을 위한 권한 설정을 먼저 진행합니다.

Bash

# 아두이노, GPS, IMU 장치 권한 부여
sudo chmod 777 /dev/ttyACM* && sudo chmod 777 /dev/ttyUSB*
🚀 실행 가이드 (Execution Guide)
각 단계는 별도의 터미널 탭에서 실행하는 것을 권장합니다.

1. 기본 시스템 및 주행 제어
시스템의 핵심인 ROS Master와 차량 제어 노드를 실행합니다.

Step 1: ROS Core 실행

Bash

roscore
Step 2: Rosserial 실행 (아두이노 통신)

Bash

source ~/catkin_ws/devel/setup.bash
rosrun rosserial_python serial_node.py
Step 3: 통합 제어 코드 실행

Bash

source ~/catkin_ws/devel/setup.bash
roslaunch henes_car_control 2024_henes_car_control.launch
2. IMU 센서 설정 (iahrs_driver)
차량의 자세(Yaw, Pitch) 데이터를 추출하고 초기화합니다.

드라이버 실행:

Bash

source ~/catkin_ws/devel/setup.bash
roslaunch iahrs_driver iahrs_driver.launch
데이터 추출 및 확인:

Yaw: rosrun iahrs_driver yaw_extractor.py (확인: rostopic echo /imu/yaw)

Pitch: rosrun iahrs_driver pitch_extractor.py (확인: rostopic echo /imu/pitch)

Yaw값 초기화:

Bash

source ~/catkin_ws/devel/setup.bash
rosservice call /euler_angle_init_cmd
3. GPS 및 UTM 좌표 변환
정밀 위치 파악을 위해 NTRIP 보정 정보를 받고 UTM 좌표로 변환합니다.

NTRIP Client 실행: python2 ~/catkin_ws/src/ntrip_ros/scripts/ntripclient-kkw1.py

RTCM 토픽 발행: python3.8 ~/catkin_ws/src/ntrip_ros/scripts/ros-ntrip-rtcm-pub.py

Ublox GPS 연결: ```bash source ~/catkin_ws/devel/setup.bash roslaunch ublox_gps ublox_device.launch

UTM 좌표 변환: ```bash source ~/catkin_ws/devel/setup.bash roslaunch gps_msgs_package gps_msgs_package.launch

* 데이터 확인: `rostopic echo /gps/utm_pos1`

4. 자율주행 및 웨이포인트
웨이포인트 기록:

Bash

source ~/catkin_ws/devel/setup.bash
rosrun my_drive waypoint_drive_with_joy.py
자율주행 모드 실행 (조향값 발행):

Bash

source ~/catkin_ws/devel/setup.bash
rosrun my_drive waypoint_drive.py
5. 비전 시스템 (Camera & Vision)
Intel RealSense 카메라를 이용해 신호등 및 장애물을 인식합니다.

카메라 드라이버 실행:

Bash

source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch enable_color:=true enable_depth:=true align_depth:=true color_width:=640 color_height:=480 color_fps:=30 depth_fps:=30
신호등 인식 (YOLOv5):

Bash

source /home/icas/catkin_ws/venvs/rosv5/bin/activate
rosrun tl_ctrl l515_yolov5_traffic_light_node.py _image_topic:=/camera/color/image_raw _yolov5_dir:=/home/$USER/catkin_ws/src/yolov5 _weights:=/home/$USER/catkin_ws/src/yolov5/yolov5s.pt _vote_window:=3 _show_debug:=true
장애물 인식:

Bash

source /home/icas/catkin_ws/venvs/rosv5/bin/activate
rosrun tl_ctrl l515_yolov5_entry_stop_node.py _image_topic:=/camera/color/image_raw _yolov5_dir:=/home/$USER/catkin_ws/src/yolov5 _weights:=/home/$USER/catkin_ws/src/yolov5/yolov5s.pt _show_debug:=true _vote_window:=3 _status_topic:=/traffic_light_state _stop_topic:=/traffic_light_st
💡 참고 사항
모든 source 명령은 사용자의 환경에 따라 ~/.bashrc에 등록하여 생략할 수 있습니다.

비전 노드 실행 시 전용 가상환경(rosv5) 활성화가 필요합니다.
