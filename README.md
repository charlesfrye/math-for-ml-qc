# Math for Machine Learning - W&B X Qualcomm

This short course introduces the core concepts and intuitions
of the three most important branches of mathematics
for machine learning:
linear algebra,
calculus,
and probability.

These notebooks are supplementary to
a lecture series (not included)
and presume proficiency with Python.

Outside of official offerings of the course,
they are intended for use in one of three ways.

## Binder - One-Click Version

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/charlesfrye/math-for-ml-qc/master)

If you click the badge above,
you'll launch a free cloud server
provided by the
[Binder project](https://mybinder.readthedocs.io/en/latest/)
with the appropriate computational environment.
This environment is _ephemeral_:
after 20 minutes of inactivity,
it will disappear.
You'll be able to run the notebooks,
but the only way to permanently save any work
is to download the files to your machine.
In order to continue from where you left off,
you'd then need to re-upload the files to a new Binder instance.

This option is simple, sufficient for most purposes
(the exercises are very short),
and well-tested.

## Local Install -- Docker

If you'd like to run the materials locally,
the best option is to use
[Docker](https://docs.docker.com/get-docker/),
one of the virtualization technologies
on which Binder is based.

After following the installation instructions,
build the container with the command
```
docker build -t math-for-ml-qc .
```
and then start it with the command
```
docker run -p 8888:8888 math-for-ml-qc
```
Open a browser window and navigate to
```
localhost:8888
```
and enter, as the password, the token that appears in the terminal after
`?token` in the URLs printed by Jupyter.

## Local Install -- pip/virtualenv

If you are unfamiliar with or unable to use Docker,
you can instead use `pip` to install the necessary packages.
They are located in `requirements-local.txt`
and can be installed with
```
pip install -r requirements-local.txt
```
Note that the requirements are very strictly versioned,
to reduce bugs.
It is highly recommended to use
a virtual environment tool,
like [virtualenv](https://virtualenv.pypa.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv)
to set up a specific environment for use with these notebooks.
In general, Python is best used with a virtual environment tool,
so setting one up will brings large dividends for future projects!
