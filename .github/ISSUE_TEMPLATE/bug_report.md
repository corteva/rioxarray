---
name: Bug report
about: Create a report to help us improve
labels: bug
---

<!-- Please search existing issues to avoid creating duplicates. -->


#### Code Sample, a copy-pastable example if possible

A "Minimal, Complete and Verifiable Example" will make it much easier for maintainers to help you:
http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports

```python
# Your code here
```
#### Problem description

[this should explain **why** the current behavior is a problem and why the expected output is a better solution.]

#### Expected Output


#### Environment Information

<!-- Only works with rioxarray >= 0.0.25 -->
 - `python -c "import rioxarray; rioxarray.show_versions()"`

<!-- For rioxarray < 0.0.25 -->

 - rioxarray version (`python -c "import rioxarray; print(rioxarray.__version__)"`)
 - rasterio version (`rio --version`)
 - GDAL version (`rio --gdal-version`)
 - Python version (`python -c "import sys; print(sys.version.replace('\n', ' '))"`)
 - Operation System Information (`python -c "import platform; print(platform.platform())"`)


#### Installation method
 - conda, pypi, from source, etc...

#### Conda environment information (if you installed with conda):

<br/>
Environment (<code>conda list</code>):
<details>

```
$ conda list | grep -E "rasterio|xarray|gdal"

```
</details>

<br/>
Details about  <code>conda</code> and system ( <code>conda info</code> ):
<details>

```
$ conda info

```
</details>
