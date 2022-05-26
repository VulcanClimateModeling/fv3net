from setuptools import setup, find_packages


with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name="fv3post",
    version="0.1.0",
    python_requires=">=3.6.0",
    author="Oliver Watt-Meyer",
    author_email="oliverwm@allenai.org",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    install_requires=requirements,
    scripts=["fv3post/scripts/fregrid_cubed_to_latlon.sh"],
    test_suite="tests",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "post_process_run=fv3post:post_process_entrypoint",
            "append_run=fv3post.append:main",
            "fregrid_single_input=fv3post.fregrid:fregrid_single_input",
        ]
    },
)
