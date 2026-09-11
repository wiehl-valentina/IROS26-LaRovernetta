import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'er_perception'

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
    description='Camera-based traversability perception (SAM-TP / GeNIE) for Earth Rover',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'traversability_node = er_perception.traversability_node:main',
            'test_image_publisher = er_perception.testing.test_image_publisher:main',
        ],
    },
)
