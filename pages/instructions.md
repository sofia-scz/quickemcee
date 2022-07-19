---
layout: sidepage
permalink: /instructions/
---

# Instructions to use the theme

Original theme is https://github.com/pages-themes/minimal

## Layouts

There are two built in layouts for markdown generated pages, one for the homepage, and another one for new pages,
such as a tutorial or a bibliography section.

For API documentation generated with pdoc3 it's necessary to create a layout file for each generated html file.

## API documentation

For convenciency save all API docs md files in a folder like docs, and all the html files in the layouts folder
with a prefix such as api- or docs-.

After the packaged is properly commented with docstrings and published in PyPI, install pdoc3 and run

```
pdoc --html -c latex_math=True packagenameinpypi
```
Remove the option latex_math if not necessary.

### Index page

pdoc3 generates an upper level directory named index. Upload the generated html to layouts, and create a markdown
file with the following

Name of the file api- or docs- index.md or something of the sorts

Contents of the file

```
---
layout: #name of the index html file uploaded in layouts without the .html
permalink: /#name of the index something like docs or api/
---
```

Additionally, go the the html file and add a return button by adding something like

```
<h3>Return to the project's <a href=" # absolute url of your project's homepage ">homepage</a></h3>
```

I prefer to add it below the line ~55 that creates the "Index" title in the sidebar, the line looks like this

```
...
<nav id="sidebar">
<h1>Index</h1>
# and here i add the line
```

### Subdirectories

And then for each subdirectory in the pdoc3 API, make a markdown page with the content

```
---
layout: #name of the html file uploaded in layouts without the .html---
permalink: /#name of the index/#name of the submodule.html
---
```

## Table of contents

There is a built in table of contents taken from https://github.com/allejo/jekyll-toc that worked out of the box by
adding the file in the included folder. By default it shows on the side bar of sidepage layout and uses the markdown
titles and subtitles to generate the section pointers.
