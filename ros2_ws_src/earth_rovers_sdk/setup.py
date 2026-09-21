import os
from glob import glob
from setuptools import setup

package_name = 'earth_rovers_sdk'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'requests', 'websocket-client', 'pygeomag'],
    zip_safe=True,
    maintainer='Earth Rover Team',
    maintainer_email='rover@frodobots.com',
    description='Earth Rovers SDK ROS2 Python nodes (bridge)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'earth_rover_bridge = earth_rovers_sdk.bridge_node:main',
            'bridge_node = earth_rovers_sdk.bridge_node:main',
        ],
    },
)
