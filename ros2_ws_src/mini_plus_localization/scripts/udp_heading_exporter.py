#!/usr/bin/env python3
import socket
import struct
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32

class UdpHeadingExporter(Node):
    def __init__(self):
        super().__init__('udp_heading_exporter')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.declare_parameter('target_ip', '127.0.0.1')
        self.declare_parameter('target_port', 9876)
        self.target_ip = self.get_parameter('target_ip').get_parameter_value().string_value
        self.target_port = self.get_parameter('target_port').get_parameter_value().integer_value

        self.subscription = self.create_subscription(
            Float32,
            'earth_rover/heading',
            self.listener_callback,
            qos_profile_sensor_data)
        self.get_logger().info(f"Exportando heading por UDP a {self.target_ip}:{self.target_port} (+ broadcast)")

    def listener_callback(self, msg: Float32):
        # Empaquetar el float32 (little-endian) y enviar
        data = struct.pack('<f', msg.data)
        try:
            self.sock.sendto(data, (self.target_ip, self.target_port))
        except Exception:
            pass
        if self.target_ip != '255.255.255.255':
            try:
                self.sock.sendto(data, ('255.255.255.255', self.target_port))
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = UdpHeadingExporter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
