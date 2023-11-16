#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rclpy                                     # ROS2 Python接口库
import socket
import time
import threading
from rclpy.node   import Node                    # ROS2 节点类
from std_msgs.msg import String                  # ROS2标准定义的String消息
from geometry_msgs.msg import PoseStamped        # ROS2标准定义的String消息


"""
创建一个订阅者节点
"""
class SubscriberNode(Node):
    def __init__(self, name):
        super().__init__(name)                             # ROS2节点父类初始化
        self.nums = 0
        self.HZ = 20                                        # 频率
        threads = []
        t1 = threading.Thread(target=self.sub_info)
        threads.append(t1)
        t2 = threading.Thread(target=self.socket_info)
        threads.append(t2)

        for t in threads:
            t.start()


    def sub_info(self):
        # 创建订阅者对象（消息类型、话题名、订阅者回调函数、队列长度）
        self.sub = self.create_subscription(PoseStamped, "/fd/ee_pose", self.listener_callback, 10)


    def listener_callback(self, msg):                               # 创建回调函数，执行收到话题消息后对数据的处理
        self.msg_data = msg
        # self.nums += 1
        # self.get_logger().info('I heard: "%s"' % self.msg_data)   # 输出日志信息，提示订阅收到的话题消息
        # time.sleep(1/self.HZ)

    def socket_info(self):
        # 创建使用IPV4(socket.AF_INET)、TCP(socket.SOCK_STREAM)协议的套接字
        tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 监听本地的8080端口
        tcp_server_socket.bind(('localhost', 8899))

        # 使用socket创建的套接字默认的属性是主动的，使用listen将其变为被动的，这样就可以接收别人的链接了
        # 开始接受连接 listen里的数字表征同一时刻能连接客户端的程度.
        tcp_server_socket.listen(128)

        while True:
            # 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
            # client_socket用来为这个客户端服务 tcp_server_socket 就可以省下来专门等待其他新客户端的链接
            # clientAddr 是元组（ip，端口）
            client_socket, clientAddr = tcp_server_socket.accept()

            t3 = threading.Thread(target=self.send_data, args=(client_socket, ))
            t3.start()


    def send_data(self, client_socket):
        while True:
            try:
                self.nums += 1

                # 向客户端发送数据
                p_x = str(self.msg_data.pose.position.x)
                p_y = str(self.msg_data.pose.position.y)
                p_z = str(self.msg_data.pose.position.z)

                o_x = str(self.msg_data.pose.orientation.x)
                o_y = str(self.msg_data.pose.orientation.y)
                o_z = str(self.msg_data.pose.orientation.z)
                o_w = str(self.msg_data.pose.orientation.w)

                pose_data = [p_x, p_y, p_z, o_x, o_y, o_z, o_w]
                pose_data = "|".join(pose_data)

                client_socket.send(pose_data.encode())

                print('Use Tcp send: "%s"' % self.nums)       # 输出日志信息，提示订阅收到的话题消息
                time.sleep(1/self.HZ)

            except ConnectionResetError as e:
                print('remote client is closed, close myself.')
                client_socket.close()
                break



def main(args=None):                                 # ROS2节点主入口main函数
    rclpy.init(args=args)                            # ROS2 Python接口初始化
    node = SubscriberNode("ee_topic_sub")            # 创建ROS2节点对象并进行初始化
    rclpy.spin(node)                                 # 循环等待ROS2退出
    node.destroy_node()                              # 销毁节点对象
    rclpy.shutdown()                                 # 关闭ROS2 Python接口
