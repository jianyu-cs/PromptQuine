import sys
import setuptools

if sys.version_info < (3, 7):
    sys.exit('Python>=3.7 is required by PromptQuine.')

# install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 by yourself
setuptools.setup(
    name="PromptQuine",
    version='0.1.0',
    url="https://github.com/jianyu-cs/PromptQuine/",
    author=("Jianyu Wang, Zhiqiang Hu, Lidong Bing"),
    description="Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='Open-Ended Prompt Evolution',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=open("requirements.txt", "r").read().split(),
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)