.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/corteva/rioxarray/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

rioxarray could always use more documentation, whether as part of the
official rioxarray docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/corteva/rioxarray/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `rioxarray` for local development.

1. Fork the `rioxarray` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/rioxarray.git

3. Create a python virtual environment

Using conda::

    $ cd rioxarray/
    $ conda env create
    $ conda activate rioxarray

Using python::

    $ cd rioxarray/
    $ python -m venv venv
    $ . venv/bin/activate

4. Install your local copy into a virtualenv.

    $ python -m pip install -e .[all]
    $ python -m pip install -r requirements/dev.txt

5. Setup pre-commit hooks::

    $ pre-commit install

6. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

7. When you're done making changes, check that the tests pass::

    $ pytest

8. Commit your changes and push your branch to GitHub (this should trigger pre-commit checks)::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

9. Submit a pull request through the GitHub website.


Running tests with docker
-------------------------

This assumes you have cloned the rioxarray repository and are in the base folder.

1. Build the docker image

.. code-block:: bash

    docker build -t rioxarray .

2. Run the tests

.. code-block:: bash

    docker run --rm \
        -v $PWD/test/:/app/test \
        -t rioxarray \
        'source /venv/bin/activate && python -m pytest'


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.10-3.12.

Tips
----

To run a subset of tests::

$ pytest test/unit/test_show_versions.py::test_get_main_info
