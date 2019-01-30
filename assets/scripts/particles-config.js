particlesJS("particles-js", {
    particles: {
      number: {
        value: 260,
        density: { enable: true, value_area: 481.0236182596568 }
      },
      color: { value: "#fff000" },
      shape: {
        type: "circle",
        stroke: { width: 0, color: "#000000" },
        polygon: { nb_sides: 5 },
        image: { src: "img/github.svg", width: 100, height: 100 }
      },
      opacity: {
        value: 1,
        random: true,
        anim: { enable: true, speed: 1, opacity_min: 0, sync: false }
      },
      size: {
        value: 3,
        random: true,
        anim: { enable: false, speed: 4, size_min: 0.3, sync: false }
      },
      line_linked: {
        enable: true,
        distance: 78.91476416322726,
        color: "#555555",
        opacity: 0.2367442924896818,
        width: 1
      },
      move: {
        enable: true,
        speed: 2,
        direction: "none",
        random: true,
        straight: false,
        out_mode: "out",
        bounce: false,
        attract: { enable: true, rotateX: 1420.4657549380909, rotateY: 600 }
      }
    },
    interactivity: {
      detect_on: "canvas",
      events: {
        onhover: { enable: true, mode: "bubble" },
        onclick: { enable: false, mode: "bubble" },
        resize: true
      },
      modes: {
        grab: {
          distance: 155.84415584415586,
          line_linked: { opacity: 0.2509491544632522 }
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
