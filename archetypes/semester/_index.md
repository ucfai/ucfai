---
{{ $semester := partial "functions/readable-semester" (dict "semester" .Name) }}

{{- $group_dir := path.Dir .File.Dir -}}
{{- $group := path.Base (path.Dir (path.Dir .File.Dir)) -}}

{{- $semesterScales := dict "fa" 2 "sp" 1 "su" 0 -}}
{{- $scale := mul 3 (index $semester "year") -}}
{{- $scale = add $scale (index $semesterScales (index $semester "shortname")) -}}
{{- $weight := sub 1000000 $scale -}}

{{- $title := printf "%s: %s" ($group | title) (index $semester "fullname") -}}

title: "{{ $title }}"
linktitle: "{{ $title }}"

{{- $semester := index $semester "semester" }}

date: <?UNK?>
location: <?UNK?>
frequency: <?UNK?>

# Summarize the Group's content for this semester
summary: >-
  We're working on filling this out!

draft: false

# DO NOT EDIT BELOW THIS LINE ----------
toc: true
weight: {{ $weight }}

menu_name: {{ $group }}_{{ $semester }}

menu:
  {{ $group }}_{{ $semester }}:
    weight: 1
  groups:
    parent: {{ $group | title }}
    identifier: {{ $group }}_{{ $semester }}

user_groups:
  - {{ printf "%s-%s-director" $semester $group }}
  - {{ printf "%s-%s-coordinator" $semester $group }}
  - {{ printf "%s-%s-guest" $semester $group }}
  - {{ printf "%s-%s-advisor" $semester $group }}
---
