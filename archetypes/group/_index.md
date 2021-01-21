---
{{ $group := .Name | title -}}
title: {{ $group }} Group
linktitle: {{ $group }}

# Optional header image (relative to `static/img/` folder).
header:
  caption: ""
  image: ""

# DO NOT MODIFY BELOW THIS LINE -------
menu_name: {{ $group }}

menu:
  main:
  groups:
  {{ $group | lower }}:
---