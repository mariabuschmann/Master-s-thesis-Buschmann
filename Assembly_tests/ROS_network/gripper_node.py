#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import pexpect

class DockerGripperController:
    def __init__(self, container_id="aea455f42660"):
        # Start a bash shell inside the container
        self.child = pexpect.spawn(f'docker exec -it {container_id} bash', encoding='utf-8', timeout=10)
        self.child.expect(['#', '\$'])  # Wait for shell prompt
        self.child.sendline('source /root/catkin_ws/devel/setup.bash')
        self.child.expect(['#', '\$'])

    def open_gripper(self):
        self.child.sendline("rosservice call /gripper_srv '{position: 60, force: 25, relative: false}'")
        self.child.expect(['#', '\$'])
        print("Gripper opened (output):", self.child.before)

    def close_gripper(self):
        self.child.sendline("rosservice call /gripper_srv '{position: 25, force: 100, relative: false}'")
        self.child.expect(['#', '\$'])
        print("Gripper closed (output):", self.child.before)

    def __del__(self):
        try:
            self.child.sendline('exit')
            self.child.close()
        except Exception:
            pass

class GripperNode(Node):
    def __init__(self):
        super().__init__('gripper_node')
        self.gripper = DockerGripperController()
        self.srv = self.create_service(SetBool, 'gripper_control', self.gripper_callback)
        self.get_logger().info("Gripper service started (True=close, False=open)")

    def gripper_callback(self, request, response):
        try:
            if request.data:
                self.gripper.close_gripper()
                response.success = True
                response.message = "Gripper closed."
            else:
                self.gripper.open_gripper()
                response.success = True
                response.message = "Gripper opened."
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
