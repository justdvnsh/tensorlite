import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorlite",
    version="0.0.1",
    author="Divyansh Dwivedi",
    author_email="justdvnsh2208@gmail.com",
    description="The lightest and minimalist edition of tensorflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justdvnsh/tensorlite",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)