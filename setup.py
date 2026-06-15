from setuptools import setup, find_packages

setup(
    name="transformer-engine-plugin-fl",
    version="0.1.0",
    description="TransformerEngine Plugin System — backend dispatch and implementations",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
)
