from setuptools import setup, find_packages

setup(
    name="concatenate_BB",
    version="0.1",
    packages=find_packages(),
    description="A custom library for concatenating audio files",
    author="Balthazar Bujard",
    author_email="balthazar.bujard@ircam.fr",
    install_requires=[
        "numpy",
        "librosa"
    ],
)