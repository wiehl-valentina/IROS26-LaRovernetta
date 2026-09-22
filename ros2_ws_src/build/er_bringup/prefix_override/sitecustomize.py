import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pablolube/IROS26-LaRovernetta/ros2_ws_src/install/er_bringup'
