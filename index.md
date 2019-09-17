---
layout: home
title:  Home
---

<img id="landing-logomark" class="pb-3"
     src="/assets/brand/fullmark/dark-bg-ready.svg"
     />

We're forging a path to turn UCF into a hotspot for researchers and engineers
passionate about computational intelligence and data science.

{% assign len = site.data.semesters[site.semester] | size %}

We host {{ len }} groups devoted to different skillsets and foci.
[Course][course] is your starting place, which feeds into
[Intelligence][intelligence] (our research group). In Fall 2019, we plan to add
[Data Science][data-science] (our industry group).

<!-- We host {{ len }} groups devoted to different skillsets and foci.
[Course][course] is your starting place, feeding into
[Data Science][data-science] (our industry group) and
[Intelligence][intelligence] (our research group). -->

[course]: {{ "/course/" | prepend: site.baseurl }}
[data-science]: {{ "/data-science/" | prepend: site.baseurl }}
[competitions]: {{ "/competitions/" | prepend: site.baseurl }}
[intelligence]: {{ "/intelligence/" | prepend: site.baseurl }}