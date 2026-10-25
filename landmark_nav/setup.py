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
    install_requires=['setuptools', 'numpy', 'pyyaml', 'opencv-python', 'ultralytics',
                      'roboticstoolbox-python', 'spatialmath-python'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='chl235@ucsd.edu',
    description='HW2: YOLO landmark detection + EKF localization + closed-loop waypoint control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = landmark_nav.detector_node:main',
            'localizer = landmark_nav.localizer_node:main',
            'closed_loop_control = landmark_nav.closed_loop_control:main',
        ],
    },
)
