from setuptools import find_packages, setup

setup(
    name="ranker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.63.0",
        "reasoner-pydantic==1.2.0.4",
        "redis==3.5.3",
        "lru-dict==1.1.6",
        "httpx==0.16.1",
        "asyncpg==0.24.0",
        "numpy==1.19.4",
        "pyyaml==5.3.1",
        "setuptools==58.0.4",
    ],
)
