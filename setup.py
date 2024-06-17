from setuptools import setup, find_packages

setup(
    name='label_AI_YOLO_augmentation',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
    ],
    author='Oleg Gerbylev',
    author_email='gerbylev.oleg@gmail.com',
    description='Библиотека для работы с изображениями',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ваш_гитхаб/my_image_lib',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)