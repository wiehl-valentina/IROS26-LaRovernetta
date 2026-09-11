import os
from glob import glob
from setuptools import setup

package_name = 'er_planning'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Earth Rover Team',
    maintainer_email='rover@frodobots.com',
    description='BEV Path Planning (GeNIE / SAM-TP) and Persistent Global Mapping for Earth Rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bev_planner_node = er_planning.bev_planner_node:main',
            'persistent_map_node = er_planning.persistent_map_node:main',
            'global_planner_node = er_planning.global_planner_node:main',
        ],
    },
)
