// The home page's 3D scene: a floodlit pitch at night. The floodlights flicker on,
// then the eleven players of the under-21 dream XI drop into their formation slots.
// Each pin links to that player's page.

import * as THREE from "../../vendor/three/three.module.js";

const TIER = { gold: 0xe6c065, silver: 0xc9d2d8, bronze: 0xc58f5f };
const PITCH_W = 68, PITCH_L = 105, MARGIN = 5;

// formation coordinates (x 0..100 across, y 0..100 up the pitch) -> world
const toWorld = (x, y) => new THREE.Vector3((x - 50) / 100 * PITCH_W * 0.68, 0, 40 - y * 0.76);

function pitchTexture() {
  const W = PITCH_W + 2 * MARGIN, L = PITCH_L + 2 * MARGIN, s = 14;   // px per metre
  const c = document.createElement("canvas");
  c.width = W * s; c.height = L * s;
  const g = c.getContext("2d");
  const stripes = 18;
  for (let i = 0; i < stripes; i++) {
    g.fillStyle = i % 2 ? "#1b6a43" : "#176040";
    g.fillRect(0, (MARGIN + (i * PITCH_L) / stripes) * s, c.width, (PITCH_L / stripes) * s + 1);
  }
  g.fillStyle = "#145638";
  g.fillRect(0, 0, c.width, MARGIN * s); g.fillRect(0, c.height - MARGIN * s, c.width, MARGIN * s);
  // grass grain
  for (let i = 0; i < 26000; i++) {
    g.fillStyle = Math.random() < 0.5 ? "rgba(255,255,255,0.025)" : "rgba(0,0,0,0.05)";
    g.fillRect(Math.random() * c.width, Math.random() * c.height, 2, 3);
  }
  g.strokeStyle = "rgba(240,246,236,0.9)";
  g.lineWidth = 0.13 * s;
  const m = (v) => (MARGIN + v) * s;
  g.strokeRect(m(0), m(0), PITCH_W * s, PITCH_L * s);
  g.beginPath(); g.moveTo(m(0), m(PITCH_L / 2)); g.lineTo(m(PITCH_W), m(PITCH_L / 2)); g.stroke();
  g.beginPath(); g.arc(m(PITCH_W / 2), m(PITCH_L / 2), 9.15 * s, 0, Math.PI * 2); g.stroke();
  for (const top of [true, false]) {
    const y0 = top ? 0 : PITCH_L, dir = top ? 1 : -1;
    g.strokeRect(m((PITCH_W - 40.32) / 2), m(top ? 0 : PITCH_L - 16.5), 40.32 * s, 16.5 * s);
    g.strokeRect(m((PITCH_W - 18.32) / 2), m(top ? 0 : PITCH_L - 5.5), 18.32 * s, 5.5 * s);
    const spot = y0 + dir * 11;
    g.beginPath();
    const a = Math.acos(5.5 / 9.15);
    if (top) g.arc(m(PITCH_W / 2), m(spot), 9.15 * s, a, Math.PI - a);
    else g.arc(m(PITCH_W / 2), m(spot), 9.15 * s, Math.PI + a, -a);
    g.stroke();
    g.fillStyle = "rgba(240,246,236,0.9)";
    g.beginPath(); g.arc(m(PITCH_W / 2), m(spot), 0.25 * s, 0, Math.PI * 2); g.fill();
  }
  g.beginPath(); g.arc(m(PITCH_W / 2), m(PITCH_L / 2), 0.3 * s, 0, Math.PI * 2); g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

function labelTexture(p, slot) {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 256;
  const g = c.getContext("2d");
  const color = `#${TIER[p.tier].toString(16).padStart(6, "0")}`;
  g.fillStyle = "rgba(8, 30, 20, 0.86)";
  roundRect(g, 46, 70, 420, 150, 26); g.fill();
  g.strokeStyle = color; g.lineWidth = 5;
  roundRect(g, 46, 70, 420, 150, 26); g.stroke();
  g.textAlign = "left"; g.textBaseline = "alphabetic";
  g.fillStyle = "#ffc94d";
  g.font = '900 112px "Big Shoulders Display", "Arial Narrow", sans-serif';
  g.fillText(String(Math.round(p.pred_t2)), 74, 182);
  g.fillStyle = "#eef3ea";
  g.font = '800 54px "Big Shoulders Display", "Arial Narrow", sans-serif';
  const name = p.name.length > 13 ? p.name.slice(0, 12) + "…" : p.name;
  g.fillText(name, 212, 140);
  g.fillStyle = "rgba(238,243,234,0.7)";
  g.font = '600 34px "Archivo", Arial, sans-serif';
  g.fillText(`${slot}  now ${p.overall}`, 214, 192);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

// soft cone of light: bright near the lamp, gone by the pitch
const beamMaterial = (color, strength) => new THREE.ShaderMaterial({
  uniforms: { uColor: { value: new THREE.Color(color) }, uStrength: { value: strength } },
  vertexShader: `varying float vY; varying vec3 vN; varying vec3 vV;
    void main(){ vY = uv.y; vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform vec3 uColor; uniform float uStrength; varying float vY; varying vec3 vN; varying vec3 vV;
    void main(){ float facing = pow(abs(dot(normalize(vN), normalize(vV))), 1.6);
      float a = pow(vY, 2.2) * uStrength * facing; gl_FragColor = vec4(uColor, a); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
});

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

export async function mountPitch(canvas, xi, { reduced = false, onPick } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(0x05170f);

  try { await document.fonts.load('900 100px "Big Shoulders Display"'); } catch { /* fall back */ }

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05170f, 0.0085);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.5, 600);
  const disposables = [];
  const track = (o) => { disposables.push(o); return o; };

  // pitch
  const pitch = new THREE.Mesh(
    track(new THREE.PlaneGeometry(PITCH_W + 2 * MARGIN, PITCH_L + 2 * MARGIN)),
    track(new THREE.MeshStandardMaterial({ map: track(pitchTexture()), roughness: 0.92, metalness: 0 })));
  pitch.rotation.x = -Math.PI / 2;
  scene.add(pitch);
  // dark ground beyond the pitch, and stands
  const ground = new THREE.Mesh(track(new THREE.PlaneGeometry(600, 600)), track(new THREE.MeshStandardMaterial({ color: 0x07160e, roughness: 1 })));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -0.05;
  scene.add(ground);
  const standMat = track(new THREE.MeshStandardMaterial({ color: 0x0b1f15, roughness: 1 }));
  const crowdPos = [];
  const stand = (w, d, x, z, rotY) => {
    const m = new THREE.Mesh(track(new THREE.BoxGeometry(w, 1, d)), standMat);
    m.position.set(x, 7, z); m.rotation.y = rotY; m.rotation.x = 0;
    m.rotateX(-0.42);
    scene.add(m);
    for (let i = 0; i < w * 3; i++) {
      const v = new THREE.Vector3((Math.random() - 0.5) * w, 0.6, (Math.random() - 0.5) * d);
      v.applyEuler(m.rotation).add(m.position);
      crowdPos.push(v.x, v.y, v.z);
    }
  };
  stand(PITCH_W + 30, 22, 0, -PITCH_L / 2 - 20, 0);
  stand(PITCH_L + 30, 22, -PITCH_W / 2 - 20, 0, Math.PI / 2);
  stand(PITCH_L + 30, 22, PITCH_W / 2 + 20, 0, -Math.PI / 2);
  const crowd = new THREE.Points(
    track(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(crowdPos, 3))),
    track(new THREE.PointsMaterial({ map: glowTexture(), color: 0xfff1cf, size: 0.45, transparent: true, opacity: 0.55, depthWrite: false })));
  scene.add(crowd);

  // light
  const hemi = new THREE.HemisphereLight(0x9ec6b0, 0x061a10, 0.35);
  scene.add(hemi);
  const towers = [[-48, -64], [48, -64], [-48, 64], [48, 64]];
  const spots = [], beams = [], lamps = [];
  const glow = track(glowTexture());
  for (const [x, z] of towers) {
    const top = new THREE.Vector3(x, 36, z);
    const pole = new THREE.Mesh(track(new THREE.CylinderGeometry(0.35, 0.6, 36, 8)), standMat);
    pole.position.set(x, 18, z);
    scene.add(pole);
    const lampMat = track(new THREE.MeshBasicMaterial({ color: 0x3a3a2a }));
    const lamp = new THREE.Mesh(track(new THREE.BoxGeometry(7, 3.2, 0.8)), lampMat);
    lamp.position.copy(top); lamp.lookAt(0, 0, 0);
    scene.add(lamp);
    const flare = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: glow, color: 0xfff0c8, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending })));
    flare.position.copy(top); flare.scale.set(22, 22, 1);
    scene.add(flare);
    lamps.push({ mat: lampMat, flare });
    const spot = new THREE.SpotLight(0xfff2d6, 0, 0, 0.62, 0.65, 0);
    spot.position.copy(top);
    spot.target.position.set(x * 0.12, 0, z * 0.12);
    scene.add(spot, spot.target);
    spots.push(spot);
    const target = new THREE.Vector3(x * 0.2, 0, z * 0.2);
    const dir = target.clone().sub(top);
    const len = dir.length();
    const beam = new THREE.Mesh(track(new THREE.ConeGeometry(26, len, 40, 1, true)), track(beamMaterial(0xfff0c8, 0)));
    beam.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir.clone().normalize());
    beam.position.copy(top).addScaledVector(dir.normalize(), len / 2);
    scene.add(beam);
    beams.push(beam);
  }

  // dust drifting through the beams
  const N = reduced ? 300 : 900;
  const dustPos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    dustPos[i * 3] = (Math.random() - 0.5) * 120;
    dustPos[i * 3 + 1] = Math.random() * 34;
    dustPos[i * 3 + 2] = (Math.random() - 0.5) * 150;
  }
  const dustGeo = track(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(dustPos, 3)));
  const dust = new THREE.Points(dustGeo, track(new THREE.PointsMaterial({ map: glow, color: 0xffe9b8, size: 0.55, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending })));
  scene.add(dust);

  // the dream XI
  const pins = [];
  const pickables = [];
  xi.forEach((s, i) => {
    const p = s.player;
    const color = TIER[p.tier] || TIER.silver;
    const group = new THREE.Group();
    const end = toWorld(s.x, s.y);
    const disc = new THREE.Mesh(track(new THREE.CircleGeometry(1.5, 40)), track(new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 })));
    disc.rotation.x = -Math.PI / 2; disc.position.y = 0.06;
    const halo = new THREE.Mesh(track(new THREE.PlaneGeometry(9, 9)), track(new THREE.MeshBasicMaterial({ map: glow, color, transparent: true, opacity: 0.0, depthWrite: false, blending: THREE.AdditiveBlending })));
    halo.rotation.x = -Math.PI / 2; halo.position.y = 0.04;
    const beam = new THREE.Mesh(track(new THREE.CylinderGeometry(0.18, 0.9, 9, 20, 1, true)), track(beamMaterial(color, 0)));
    beam.position.y = 4.5; beam.rotation.x = Math.PI;
    const label = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: track(labelTexture(p, s.slot)), transparent: true, opacity: 0, depthWrite: false })));
    label.scale.set(9, 4.5, 1); label.position.y = 10.5;
    label.userData.id = p.id; disc.userData.id = p.id;
    group.add(halo, disc, beam, label);
    const start = new THREE.Vector3((Math.random() - 0.5) * 90, 40 + Math.random() * 25, -40 - Math.random() * 50);
    group.position.copy(reduced ? end : start);
    scene.add(group);
    pins.push({ group, start, end, beam, halo, label, delay: 1.0 + i * 0.09, landed: reduced });
    pickables.push(label, disc);
  });

  // interaction
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2(9, 9);
  const parallax = { x: 0, y: 0 };
  let hovered = null;
  const onMove = (e) => {
    const r = canvas.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    parallax.x = pointer.x; parallax.y = pointer.y;
  };
  const onLeave = () => { pointer.set(9, 9); parallax.x = parallax.y = 0; };
  const onClick = () => { if (hovered && onPick) onPick(hovered); };
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  canvas.addEventListener("click", onClick);

  // render loop, paused when the hero is off screen or the tab is hidden
  // the intro runs on wall-clock time so slow machines still see it finish on schedule
  let raf = 0, running = false, t = reduced ? 10 : 0, last = 0;
  // keep the pitch clear of the headline: on wide screens the scene is shifted right of
  // the copy, on narrow ones it is lifted above it
  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const wide = w >= 900;
    camera.fov = wide ? 38 : 58;
    if (wide) camera.setViewOffset(w, h, -w * 0.225, h * 0.02, w, h);
    else camera.setViewOffset(w, h, 0, h * 0.06, w, h);
    camera.updateProjectionMatrix();
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  const lightLevel = (time) => {
    // the floodlights warm up with a couple of flickers
    if (time < 0.15) return 0;
    if (time < 0.3) return 0.55;
    if (time < 0.42) return 0.1;
    if (time < 0.55) return 0.8;
    if (time < 0.62) return 0.35;
    return Math.min(1, 0.6 + (time - 0.62) * 1.2);
  };

  function frame(now) {
    const dt = last ? Math.min((now - last) / 1000, 0.25) : 0;
    last = now;
    t += dt;
    const L = reduced ? 1 : lightLevel(t);
    spots.forEach((s) => (s.intensity = 2.6 * L));
    beams.forEach((b) => (b.material.uniforms.uStrength.value = 0.075 * L));
    lamps.forEach(({ mat, flare }) => { mat.color.setRGB(0.2 + 0.8 * L, 0.2 + 0.75 * L, 0.18 + 0.6 * L); flare.material.opacity = 0.9 * L; });
    hemi.intensity = 0.15 + 0.3 * L;
    dust.material.opacity = 0.55 * L;
    crowd.material.opacity = 0.2 + 0.4 * L;

    for (const pin of pins) {
      const k = reduced ? 1 : Math.max(0, Math.min(1, (t - pin.delay) / 1.3));
      const e = easeOut(k);
      if (!pin.landed) {
        pin.group.position.lerpVectors(pin.start, pin.end, e);
        pin.group.position.y = pin.end.y + (1 - e) * pin.start.y + Math.sin(e * Math.PI) * 6;
        if (k >= 1) pin.landed = true;
      }
      const after = reduced ? 1 : Math.max(0, Math.min(1, (t - pin.delay - 1.1) / 0.6));
      pin.beam.material.uniforms.uStrength.value = 0.55 * after;
      pin.halo.material.opacity = 0.55 * after;
      pin.label.material.opacity = after;
    }

    if (!reduced) {
      const pos = dustGeo.attributes.position;
      for (let i = 0; i < N; i++) {
        let y = pos.array[i * 3 + 1] + dt * 0.35;
        if (y > 34) y = 0;
        pos.array[i * 3 + 1] = y;
      }
      pos.needsUpdate = true;
    }

    const sway = -0.05 + (reduced ? 0 : Math.sin(t * 0.08) * 0.07);
    const a = sway + parallax.x * 0.06;
    const R = 74;
    const intro = reduced ? 1 : easeOut(Math.min(1, t / 3.2));
    camera.position.set(Math.sin(a) * R, 44 + (1 - intro) * 16 + parallax.y * 3, Math.cos(a) * R + (1 - intro) * 16);
    camera.lookAt(0, 0, 2);

    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickables, false)[0];
    const id = hit ? hit.object.userData.id : null;
    if (id !== hovered) {
      hovered = id;
      canvas.style.cursor = id ? "pointer" : "";
      for (const pin of pins) {
        const on = pin.label.userData.id === id;
        pin.label.scale.set(on ? 10.4 : 9, on ? 5.2 : 4.5, 1);
      }
    }
    renderer.render(scene, camera);
    if (running) raf = requestAnimationFrame(frame);
  }
  const start = () => { if (!running) { running = true; last = 0; raf = requestAnimationFrame(frame); } };
  const stop = () => { running = false; cancelAnimationFrame(raf); };
  let onScreen = true;
  const io = new IntersectionObserver(([e]) => { onScreen = e.isIntersecting; onScreen && !document.hidden ? start() : stop(); });
  io.observe(canvas);
  const onVis = () => (document.hidden || !onScreen ? stop() : start());
  document.addEventListener("visibilitychange", onVis);
  start();

  return {
    dispose() {
      stop();
      io.disconnect(); ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("click", onClick);
      for (const d of disposables) d.dispose?.();
      renderer.dispose();
    },
  };
}
