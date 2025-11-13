from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'langgraph_base_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('lib/' + package_name, [package_name + '/langgraph_base.py']),
        ('lib/' + package_name, [package_name + '/langgraph_ros_base.py']),
        ('lib/' + package_name, [package_name + '/ollama_utils.py']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.json'))),
        (os.path.join('share', package_name, 'params'),
            glob(os.path.join('params', '*.yaml')))
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=False,
    maintainer='Oscar Pons Fernandez',
    maintainer_email='opfernandez@uma.es',
    description='ROS2 langgraph base classes for heterogeneous use cases implementation',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # entry_points={
    #     'console_scripts': [
    #         'langgraph_base_ros = ' + package_name + '.langgraph_base_ros:main',
    #     ],
    # },
)
