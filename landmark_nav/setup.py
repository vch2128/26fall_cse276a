from glob import glob

from setuptools import find_packages, setup

package_name = 'landmark_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml', 'opencv-python'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='chl235@ucsd.edu',
    description='AprilTag detection + EKF localization in a known map; HW4 path planning (Voronoi / RRT*) + control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = landmark_nav.detector_node:main',
            'localizer = landmark_nav.localizer_node:main',
            'planner = landmark_nav.planner:main',
            'hw4_control = landmark_nav.hw4_control:main',
        ],
    },
)
