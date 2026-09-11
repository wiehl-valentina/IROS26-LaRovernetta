import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'er_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Earth Rover Team',
    maintainer_email='rover@frodobots.com',
    description='GPS Waypoint Controller for Earth Rover Mission 1',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gps_waypoint_controller = er_navigation.gps_waypoint_controller:main',
            'test_nav_stimulus = er_navigation.testing.test_nav_stimulus:main',
            'test_dual_heading_stimulus = er_navigation.testing.test_dual_heading_stimulus:main',
            'heading_diagnostic_monitor = er_navigation.testing.heading_diagnostic_monitor:main',
            'run_heading_conflict_experiment = er_navigation.testing.run_heading_conflict_experiment:main',
            'diagnose_yaw_sign = er_navigation.testing.diagnose_yaw_sign:main',
        ],
    },
)
