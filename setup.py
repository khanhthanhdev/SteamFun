from setuptools import setup, find_packages

setup(
    name="video-generation-agents",
    version="0.1.0",
    description="Multi-agent video generation system with LangGraph and AWS integration",
    author="Video Generation Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "langgraph>=0.2.0",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-anthropic>=0.2.0",
        "langchain-community>=0.3.0",
        "langchain-core>=0.3.0",
        "boto3>=1.34.0",
        "botocore>=1.34.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.8.0",
        "asyncio",
        "aiofiles",
        "pillow",
        "opencv-python-headless",
        "numpy",
        "requests",
        "httpx",
        "uvicorn",
        "fastapi",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "manim": [
            "manim>=0.18.0",
        ],
        "aws": [
            "aioboto3>=12.0.0",
            "aiobotocore>=2.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "video-gen=src.langgraph_agents.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)