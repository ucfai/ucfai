# ucfai.org
**Hugo Rewrite edition**

## Installation and setup
1. Clone this repository, using something like...
   `git clone git@github.com:ucfai/ucfai.github.io`.
1. Checkout the `hugo-migration` branch, this is what we'll be working from.
1. Make sure you run `git submodule update --init --recursive` to properly
   "install" `hugo-academic`.
1. From the repository root, run: `docker-compose up -d`.

## Interaction with the Docker Container
We're using `docker-compose` to simplify and standardize the development/build
process. Because of this, you need to become semi-familiar with the
`docker-compose` CLI interface.

**To start the container:**

```bash
docker-compose up
```
This will start the container and attach the log output to your current terminal window.


**To start the container in a `detached` state:**

```bash
docker-compose up -d
```
This will start the container in a `detached` state, which allows it to persist
(continue running) beyond the terminal window. All it requires is that `docker`
is running on your host machine. 

**To access the log from this container:**
```bash
docker-compose logs hugo-ucfai-org
```
As this container is named, you'll only be able to spawn a single instance of it
(which makes it trivial to access the logs, too). As a build system, `hugo` dumps
all errors to this log, so you'll be able to catch a glimpse of what's going on
by running the above command.

**Personal recommendation:** Run `docker-compose up` whenever you're developing,
the latter methods of running are useful, but can make it a bit difficult to
track down errors until you've gotten into the headspace work with `docker`,
which takes a bit.
