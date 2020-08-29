particlesJS("particles-js", {
    particles: {
      number: {
        value: 260,
        density: { enable: true, value_area: 481 }
      },
      color: { value: "#fff000" },
      shape: {
        type: "circle",
        stroke: { width: 0, color: "#000000" },
        polygon: { nb_sides: 5 },
        image: { src: "img/github.svg", width: 100, height: 100 }
      },
      opacity: {
        value: 0.8,
        random: true,
        anim: { enable: true, speed: 1, opacity_min: 0.1, sync: false }
      },
      size: {
        value: 4,
        random: true,
        anim: { enable: true, speed: 4, size_min: 0.4, sync: false }
      },
      line_linked: {
        enable: true,
        distance: 78,
        color: "#fdde3c",
        opacity: 0.20,
        width: 1
      },"detect_on": "window",
      move: {
        enable: true,
        speed: 2,
        direction: "none",
        random: true,
        straight: false,
        out_mode: "out",
        bounce: false,
        attract: { enable: true, rotateX: 1420, rotateY: 600 }
      }
    },
    interactivity: {
      detect_on: "window",
      events: {
        onhover: { enable: true, mode: "grab" },
        onclick: { enable: false, mode: "bubble" },
        resize: true
      },
      modes: {
        grab: {
          distance: 100,
          line_linked: { opacity: 0.60 }
        },
        bubble: {
          distance: 146.17389821424212,
          size: 10,
          duration: 2,
          opacity: 0,
          speed: 3
        },
        repulse: { distance: 400, duration: 0.4 },
        push: { particles_nb: 4 },
        remove: { particles_nb: 2 }
      }
    },
    retina_detect: true
  });
