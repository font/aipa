from setuptools import setup, find_packages

setup(
    name="aipa",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llama-index>=0.9.0",
        "llama-stack-client>=0.1.0",
        "fastapi>=0.103.1",
        "uvicorn>=0.23.2",
        "python-dotenv>=1.0.0",
        "langchain>=0.0.267",
        "pydantic>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "aipa=src.__main__:main",
        ],
    },
) 