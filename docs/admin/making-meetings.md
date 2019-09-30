---
title: Building Your Meetings
layout: page
---
The bot (`autobot`) generates the meeting notebooks based on the details
specified in this Group's semester `syllabus.yml`. To make changes to the
`syllabus.yml`, take a look at [the documentation][edit-syllabus].

There are at least 3 components to a meeting:
- (make edits to this) [`YYYY-MM-DD-<filename>.solution.ipynb`][notebook-solution]
- (auto-generated) [`YYYY-MM-DD-<filename>.ipynb`][notebook-meeting]
- (auto-generated) [`kernel-metadata.json`][kernel-metadata]

[notebook-solution]: #notebook-solution
[notebook-meeting]:  #notebook-meeting
[kernel-metadata]:   #kernel-metadata

<br />

<span id="notebook-solution"></span>
## The Solution Notebook
**`YYYY-MM-DD-<filename>.solution.ipynb`**

Notebooks should be built according to your Group's standards. However,
some general ideas apply.

### Jupyter makes use of [Twitter Bootstrap][bootstrap]
To add questions or otherwise generate text which shouldn't show up as plain
text in a document, consider making use of Bootstrap Alerts, like:
```html
<div class="alert alert-warning">
  <h4> This is a Twitter Bootstrap alert header </h4>
  <p class="m-0"> And some text which can be used for further explanation. </p>
</div>
```
<div class="alert alert-warning">
  <h4> This is a Twitter Bootstrap alert header </h4>
  <p class="m-0"> And some text which can be used for further explanation. </p>
</div>

### Adding Solution Cells
Solution notebooks are akin to the solution manuals faculty have for their
textbooks. As a result, we've made use of some tools to allow us to provide
solutions for students while also giving them an opportunity to solve the
notebooks themselves.

Currently, we **can't** set variables of any kind for solutions, e.g.
```python
some_col = ... # YOUR SOLUTION HERE
```
**instead**, one should detail similar code "fill-in-the-blanks" like so:
```python
# select a column from the `test_df` DataFrame by setting `some_col` to one of
#   the column names, like so:
# some_col = "unknown-variable"
### BEGIN SOLUTION
some_col = "known-variable"
### END SOLUTION
```
Upon adding solution blocks to the cell, you'll see it replaced (in the meeting
notebook) with:
```python
# select a column from the `test_df` DataFrame by setting `some_col` to one of
#   the column names, like so:
# some_col = "unknown-variable"
# YOUR CODE HERE
raise NotImplementedError()
```
In the event you're using a class, like when imeplementing a neural network with
`nn.Module`.
```python
class VanillaNetwork(nn.Module):
    def __init__(self):
        ### BEGIN SOLUTION
        # ... implementation network layers
        ### END SOLUTION

    def forward(self, x):
        ### BEGIN SOLUTION
        # ... implementation of the foward pass
        ### END SOLUTION
```
This will do similarly to what was shown above:
```python
class VanillaNetwork(nn.Module):
    def __init__(self):
        # YOUR CODE HERE
        raise NotImplementedError()

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError()
```

<br />

<span id="notebook-meeting"></span>
## The Meeting Notebook
**`YYYY-MM-DD-<filename>.solution.ipynb`**

**Do not create, edit, or otherwise touch this Notebook.** `autobot** will
automatically generate this with a full overwrite every time.

<br />

<span id="kernel-metadata"></span>
## Setting Kernel Metadata
**`kernel-metadata.json`**

As we've transitioned to using Kaggle Kernels to run our meetings, every
meeting has an accompanying `kernel-metadata.json` which Kaggle needs to allow
for programmatic upload. Below is an example `kernel-metadata.json`.
```json
{
    "id": "ucfaibot/core-fa19-regression",
    "title": "core-fa19-regression",
    "code_file": "2019-09-18-regression.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": false,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": ["uciml/adult-census-income"],
    "competition_sources": ["ucfai-core-fa19-regression", "house-prices-advanced-regression-techniques"],
    "kernel_sources": []
}
```
To make changes to this `kernel-metadata.json`, you must edit the
`syllabus.yml` for your corresponding lecture. **There's no need to specify
the notebook link on kaggle, nor the competition for a given meeting. `autobot`
will do that for you.**
```yaml
# ...
optional:
  kaggle:
    datasets: []
    competitions: []
    kernel: []
```
Follow the instructions from the [Kaggle API Wiki][kaggle-metadata-api] to
understand the formatting to specify `datasets`, `competitions`, and
`kernels` to pull from.

[edit-syllabus]: https://ucfai.org/docs/admin/seed-semester#syllabus-yml
[bootstrap]: https://bootstrap.com/
[kaggle-metadata-api]: https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata
