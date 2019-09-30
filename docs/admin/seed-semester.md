---
title: Seeding a Group's Semester
layout: page
---
Semester's consist of 3 documents:
1. [`env.yml`](#env-yml)
1. [`overhead.yml`](#overhead-yml)
1. [`syllabus.yml`](#syllabus-yml)

<span id="env-yml"></span>

## Anaconda environment file
**`env.yml`**

This is largely seeded from the original semester in which the `bot` was built.
Overall, it acts just like any other Anaconda environment, but has been
pre-named for consistency between semesters. **To make changes, just edit this
like any other Anaconda environment file.**

<br />

<span id="overhead-yml"></span>
## Semester's "metadata"
**`overhead.yml`**

The semester generally has some associated "metadata** which is generally tied to
every meeting, e.g. meeting days and times, coordinators, etc.

**Adding Coordinators:**
For example, adding Coordinators, one should specify their GitHub username and
their particular role in the Group. Don't concern yourself with "rank" ordering
as the locations this is copied to account for that (e.g. on the website).
```yaml
coordinators:
- github: "ionlights"
  role: "Director"
```

**Setting Meeting Metadata:**
Specify the `start_offset`, `wday`, `time`, and `room` for the Group.
- `start_offset` &ndash; the week number of the semester on which the Group
   starts.
- **`time`** &ndash; specified in military time, without `:`, e.g. `1830-2030`.
- **`wday`** &ndash; the 3-letter acronym of whatever weekday.
- **`room`** &ndash; building code and room number

```yaml
meetings:
  start_offset: 3
  wday: "Wed"
  time: "1730-1930"
  room: "HEC 101"
```

<br />

<span id="overhead-yml"></span>
## Semester's Syllabus
**`syllabus.yml`**

Below is an example syllabus entry. As marked, `required` sections will throw
errors if unfilled. `optional` sections are present even if "irrelevant" to a
given group. For example, `papers` are ulikely to be filled in `core`, as such
&ndash; it's unneccesary to fill it out.
```yaml
- required:
    title: "Welcome back! Introducing Data Science"
    filename: "welcome-back"
    cover: "https://www.autodesk.com/products/eagle/blog/wp-content/uploads/2018/04/shutterstock_1011096853.jpg"
    instructors: ["sirroboto"]
    description: >-
      Welcome back or welcome aboard to AI@UCF! We'll be covering what we do,
      how we do it, and what opportunities are available for you. Food and drink
      will be provided, so take the time to eat, chat, and learn about our club.
      See you there!
  optional:
    date: ""
    tags: ["club"]
    slides: ""
    papers: []
    kaggle:
      datasets: []
      competitions: []
      kernels: []
```

**`required` items**
- **`title`** &ndash; a human-readable name of the meeting, should be less than 50
  characters.
- **`filename`** &ndash; a URL slug for the meeting, take the example:
   https://ucfai.org/core/fa18/neural-nets/, the slug is the `neural-nets` bit.
   This should be concise, yet readable and descriptive of the meeting. *Must be
   unique to a semester*
- **`cover`** &ndash; a link to an image which can be used in the meeting banner.
   This image should be at least 950px in height.
- **`instructors`** &ndash; array to support multiple instructors, use their
  `github` usernames; case-insensitive.
- **`description`** &ndash; long-form description of the meeting.

**`optional` items**
- **`date`** &ndash; only on events which are intermittent, e.g. Supplementary
  meetings which meet on regular days, but irregularly. Meaning they meet twice
  in a row, with a 3 week break, then once again (as a meeting pattern).
- **`room`** &ndash; in the event a different room than regular must be used.
- **`tags`** &ndash; descriptive, yet concise, tags for a given meeting.
- **`slides`** &ndash; a link to the google slides, provide a link from the "URL
  Sharing" option on Google Slides.
- **`papers`** &ndash; an array which has direct links to a downloadable PDF. The
  bot will handle downloading and renaming the papers appropriately.
- **`kaggle`** &ndash; Follow the instructions from the
  [Kaggle API Wiki][kaggle-metadata-api] to understand the formatting to specify
  `datasets`, `competitions`, and `kernels` to pull from.
  - **`datasets`** &ndash; additional data sources from Kaggle.
  - **`competitions`** &ndash; competitoions which we can use as data sources as,
    as well as additional submission sources. **There's no need to add the
    competitions which are hosted by us. `autobot` handles this for you.**
  - **`kernel`** &ndash; additional code sources from Kaggle.

[kaggle-metadata-api]: https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata
