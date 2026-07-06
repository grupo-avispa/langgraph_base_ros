"""Shared pytest fixtures and setup for the langgraph_base_ros test suite."""

import sys
from unittest.mock import MagicMock

try:
    import rclpy.node  # noqa: F401
except ModuleNotFoundError:
    # Some CI environments run these unit tests without a full ROS 2 install
    # (rclpy is unavailable on PYTHONPATH). Since the tests never call
    # rclpy.init() and always mock away Node's ROS-specific methods, a plain
    # stand-in class is enough to satisfy the `class LangGraphRosBase(Node)`
    # import-time inheritance.
    class _FakeNode:
        """Stand-in for rclpy.node.Node used only when rclpy is not installed."""

        def __init__(self, *args, **kwargs):
            pass

        def get_logger(self):
            pass

        def declare_parameter(self, *args, **kwargs):
            pass

        def get_parameter(self, *args, **kwargs):
            pass

    rclpy_stub = MagicMock(name='rclpy')
    rclpy_node_stub = MagicMock(name='rclpy.node')
    rclpy_node_stub.Node = _FakeNode
    rclpy_stub.node = rclpy_node_stub

    sys.modules['rclpy'] = rclpy_stub
    sys.modules['rclpy.node'] = rclpy_node_stub
