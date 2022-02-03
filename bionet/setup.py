from setuptools import setup, find_packages

setup(
    name='bionet',
    version='0.0.1',    
    description='bionet implementation',
    url='https://github.com/',
    author='xxx',
    author_email='xxx',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['torch>=1.5','torchvision>=0.8','sklearn'],
	python_requires=">=3",

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)