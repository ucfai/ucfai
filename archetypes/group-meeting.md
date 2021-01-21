---
{{ $semesterAbbrs := dict "fa" "Fall" "sp" "Spring" "su" "Summer" -}}
{{ $semester := path.Base (path.Dir .File.Dir) -}}
{{ $shortname := substr $semester 0 2 -}}
{{ $year := int (substr $semester 2 2) -}}
{{ $fullname := printf "%s %s" (index $semesterAbbrs $shortname) (string (add 2000 $year))}}

{{- $group_dir := path.Dir (path.Dir .File.Dir) -}}
{{- $group := path.Base $group_dir -}}

title: "<?UNK?>"
linktitle: "<?UNK?>"

date: "<?UNK?>"
lastmod: "<?UNK?>"

draft: false
toc: true

weight: 0

menu:
  {{ $group }}_{{ $semester }}:
    parent: {{ $fullname }}

authors: []

papers: {}

location: ""
cover: ""

categories: ["{{ $semester }}"]
tags: []
abstract: >-

---