"""Optional Colab point picker. No Colab dependency is imported by the core pipeline."""

import base64
import json

import cv2

from .geometry import CalibrationSpec


def named_landmarks(length=40, width=20):
    points = {
        "corner_left_far": [0, 0], "corner_right_far": [length, 0],
        "corner_right_near": [length, width], "corner_left_near": [0, width],
        "halfway_far": [length / 2, 0], "halfway_near": [length / 2, width],
        "centre": [length / 2, width / 2],
        "circle_far": [length / 2, width / 2 - 3],
        "circle_near": [length / 2, width / 2 + 3],
        "circle_left": [length / 2 - 3, width / 2],
        "circle_right": [length / 2 + 3, width / 2],
    }
    for side, x, sign in (("left", 0, 1), ("right", length, -1)):
        points[f"penalty_{side}"] = [x + sign * 6, width / 2]
        points[f"second_penalty_{side}"] = [x + sign * 10, width / 2]
        for position, y, edge in (("far", width / 2 - 1.58, -1),
                                   ("near", width / 2 + 1.58, 1)):
            points[f"arc_{side}_{position}_baseline"] = [x, y + edge * 6]
            points[f"arc_{side}_{position}_tangent"] = [x + sign * 6, y]
    return points


def pick_calibration(image, time_s, length=40, width=20):
    """Choose named floor landmarks and independent checks in a Colab output canvas.

    Far/near orientation refers to the rendered diagram, not an inferred camera orientation.
    Rotate the court interpretation consistently if the camera faces the opposite side.
    Returns an UNREVIEWED specification; the caller must explicitly review correspondences.
    """
    from google.colab.output import eval_js

    _, encoded = cv2.imencode(".jpg", image)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    points = named_landmarks(length, width)
    template = r"""(async () => {
      const image = new Image(); image.src = IMAGE_URI;
      await new Promise(resolve => image.onload = resolve);
      const root = document.createElement('div'); document.body.appendChild(root);
      const instructions = document.createElement('p');
      instructions.textContent = 'Select a named floor landmark, then click its exact image position. Use 4+ fit points and 1+ different check point. Never click a raised goalpost.';
      root.appendChild(instructions);
      const names = LANDMARKS;
      const select = document.createElement('select');
      Object.entries(names).forEach(([name, xy]) => {
        const option = document.createElement('option'); option.value = name;
        option.textContent = name + ' (' + xy.join(', ') + ' m)'; select.appendChild(option);
      }); root.appendChild(select);
      const check = document.createElement('input'); check.type = 'checkbox'; root.appendChild(check);
      root.appendChild(document.createTextNode('Independent validation point '));
      const undo = document.createElement('button'); undo.textContent = 'Undo'; root.appendChild(undo);
      const done = document.createElement('button'); done.textContent = 'Finish'; root.appendChild(done);
      const canvas = document.createElement('canvas');
      const scale = Math.min(1, 1100 / image.width); canvas.width = image.width * scale;
      canvas.height = image.height * scale; canvas.style.display = 'block'; root.appendChild(canvas);
      const ctx = canvas.getContext('2d'), selected = [];
      function draw() {
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        selected.forEach((p, i) => {
          ctx.fillStyle = p.validation ? '#ffe04b' : '#ff5555'; ctx.beginPath();
          ctx.arc(p.image[0]*scale, p.image[1]*scale, 5, 0, 2*Math.PI); ctx.fill();
          ctx.font = '14px sans-serif'; ctx.fillText(i+': '+p.name, p.image[0]*scale+8, p.image[1]*scale-8);
        });
      }
      canvas.onclick = event => {
        const rect = canvas.getBoundingClientRect();
        if(selected.some(p => p.name === select.value)) { alert('Choose a different landmark.'); return; }
        selected.push({name: select.value, court: names[select.value], validation: check.checked,
          image: [(event.clientX-rect.left)*image.width/rect.width,
                  (event.clientY-rect.top)*image.height/rect.height]}); draw();
      };
      undo.onclick = () => {selected.pop(); draw();}; draw();
      return await new Promise(resolve => done.onclick = () => {
        if(selected.filter(p=>!p.validation).length < 4 || !selected.some(p=>p.validation)) {
          alert('Need at least 4 fit landmarks and one independent validation landmark.'); return;
        }
        resolve(selected);
      });
    })()"""
    script = template.replace("IMAGE_URI", json.dumps(data_uri)).replace("LANDMARKS", json.dumps(points))
    selected = eval_js(script)
    fit = [p for p in selected if not p["validation"]]
    check = [p for p in selected if p["validation"]]
    return CalibrationSpec(time_s=time_s, length=length, width=width,
                           image_points=[p["image"] for p in fit], court_points=[p["court"] for p in fit],
                           validation_image_points=[p["image"] for p in check],
                           validation_court_points=[p["court"] for p in check])
