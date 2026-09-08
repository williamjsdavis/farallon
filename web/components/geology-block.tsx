'use client';

import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { RockUnit, Volume } from '@/lib/geology-types';

type SceneState = {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  block: THREE.Group;
};
function webglUnavailable(container: HTMLDivElement) {
  const message = document.createElement('p');
  message.className = 'webgl-error';
  message.textContent =
    'WebGL is unavailable. The geological maps and hypothesis loop still work.';
  container.replaceChildren(message);
  return () => message.remove();
}
function disposeGroup(group: THREE.Group) {
  group.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    if (mesh.material)
      for (const material of Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material]) {
        (material as THREE.MeshBasicMaterial).map?.dispose();
        material.dispose();
      }
  });
  group.clear();
}

export function GeologyBlock({
  volume,
  surfaceImage,
  palette,
  cut,
}: {
  volume: Volume;
  surfaceImage: string;
  palette: RockUnit[];
  cut: number;
}) {
  const host = useRef<HTMLDivElement>(null),
    state = useRef<SceneState | null>(null);
  const bytes = useMemo(
    () => Uint8Array.from(atob(volume.data), (char) => char.charCodeAt(0)),
    [volume.data],
  );
  useEffect(() => {
    if (!host.current) return;
    const container = host.current;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      return webglUnavailable(container);
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);
    const scene = new THREE.Scene(),
      camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
    camera.position.set(7.6, 5.9, 8.4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.55, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4;
    controls.maxDistance = 23;
    controls.maxPolarAngle = Math.PI * 0.87;
    const block = new THREE.Group();
    scene.add(block);
    const grid = new THREE.GridHelper(16, 16, 0x33424c, 0x25313a);
    grid.position.y = -1.83;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.48;
    scene.add(grid);
    state.current = { scene, renderer, camera, controls, block };
    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      renderer.setSize(width, height);
      camera.aspect = width / Math.max(1, height);
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();
    let frame = 0;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    animate();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      disposeGroup(block);
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, []);

  useEffect(() => {
    const context = state.current;
    if (!context || !volume) return;
    const { block } = context;
    disposeGroup(block);
    let disposed = false;
    const [nz, ny, nx] = volume.shape;
    const [xmin, xmax, ymin, ymax, zmin, zmax] = volume.bounds;
    const startY = Math.min(ny - 2, Math.floor(cut * ny));
    const fraction = startY / ny,
      ycut = ymin + (ymax - ymin) * fraction;
    const cx = (xmin + xmax) / 2,
      cy = (ymin + ymax) / 2;
    const colors = new Uint8Array(256 * 4);
    palette.forEach((unit) => colors.set([...unit.rgb, 255], unit.id * 4));
    const voxel = (x: number, y: number, z: number) =>
      bytes[(z * ny + y) * nx + x];
    function texture(
      width: number,
      height: number,
      sample: (x: number, y: number) => number,
    ) {
      const rgba = new Uint8Array(width * height * 4);
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          const id = sample(x, y);
          rgba.set(colors.subarray(id * 4, id * 4 + 4), (y * width + x) * 4);
        }
      const map = new THREE.DataTexture(rgba, width, height, THREE.RGBAFormat);
      map.colorSpace = THREE.SRGBColorSpace;
      map.magFilter = THREE.NearestFilter;
      map.minFilter = THREE.NearestFilter;
      map.needsUpdate = true;
      return map;
    }
    function face(corners: number[][], map: THREE.Texture, shade = 0xffffff) {
      const positions = corners.flatMap(([x, y, z]) => [x - cx, z, -(y - cy)]);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(positions, 3),
      );
      geometry.setAttribute(
        'uv',
        new THREE.Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2),
      );
      geometry.setIndex([0, 1, 2, 0, 2, 3]);
      geometry.computeVertexNormals();
      block.add(
        new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            map,
            color: shade,
            side: THREE.DoubleSide,
            transparent: true,
            alphaTest: 0.1,
          }),
        ),
      );
      block.add(
        new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry),
          new THREE.LineBasicMaterial({
            color: 0xdce6e8,
            transparent: true,
            opacity: 0.25,
          }),
        ),
      );
    }
    // PNG rows run north to south. Standard Texture.flipY gives north at v=1.
    const surface = new THREE.TextureLoader().load(surfaceImage, (map) => {
      if (disposed) map.dispose();
    });
    surface.colorSpace = THREE.SRGBColorSpace;
    surface.magFilter = THREE.NearestFilter;
    surface.minFilter = THREE.NearestFilter;
    surface.offset.y = fraction;
    surface.repeat.y = 1 - fraction;
    face(
      [
        [xmin, ycut, zmax],
        [xmax, ycut, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
      ],
      surface,
    );
    face(
      [
        [xmin, ycut, zmin],
        [xmax, ycut, zmin],
        [xmax, ycut, zmax],
        [xmin, ycut, zmax],
      ],
      texture(nx, nz, (x, z) => voxel(x, startY, z)),
      0xe6ebf0,
    );
    face(
      [
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
      ],
      texture(nx, nz, (x, z) => voxel(x, ny - 1, z)),
      0xd0d9e2,
    );
    face(
      [
        [xmin, ycut, zmin],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmin, ycut, zmax],
      ],
      texture(ny - startY, nz, (y, z) => voxel(0, y + startY, z)),
      0xc7d1db,
    );
    face(
      [
        [xmax, ycut, zmin],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax],
        [xmax, ycut, zmax],
      ],
      texture(ny - startY, nz, (y, z) => voxel(nx - 1, y + startY, z)),
      0xc7d1db,
    );
    face(
      [
        [xmin, ycut, zmin],
        [xmax, ycut, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
      ],
      texture(nx, ny - startY, (x, y) => voxel(x, y + startY, 0)),
      0xc0cad4,
    );
    return () => {
      disposed = true;
    };
  }, [volume, surfaceImage, palette, cut, bytes]);
  return (
    <div
      className="geology-canvas"
      ref={host}
      aria-label="Interactive 3D geological block. Drag to orbit and use the cutaway slider to reveal its interior."
    />
  );
}
