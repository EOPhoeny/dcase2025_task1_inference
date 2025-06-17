from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='Zhang_AITHU_SJTU_task1',
    version='0.1.0',
    description='SSCP-Mobile Inference package',
    author='Shuwei Zhang, Bing Han',
    author_email="eophoeny@gmail.com",
    packages=find_packages(),  # This auto-discovers the inner folder
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'Zhang_AITHU_SJTU_task1': ["resources/*.wav", 'ckpts/*.ckpt'],
    },
    python_requires='>=3.10',
)
