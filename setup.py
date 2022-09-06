from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='job_prepr_model',
      version="0.0.1",
      description="Job Prepr (CNN model)",
      license="MIT",
      author="Le Wagon 960 DreamTeam",
      author_email="contact@jobprepr.bot",
      url="https://github.com/skoooty/job-prepr-model",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
