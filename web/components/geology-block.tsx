'use client';

import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { RockUnit, Volume } from '@/lib/geology-types';

type SceneState = {
  renderer: THREE.WebGLRenderer;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  block: THREE.Group;
  grid: THREE.GridHelper;
  surfaceTexture: THREE.Texture | null;
  surfaceImage: string | null;
};
function webglUnavailable(container: HTMLDivElement) {
  const message = document.createElement('p');
  message.className = 'webgl-error';
  message.textContent =
    'WebGL is unavailable. The geological maps and hypothesis loop still work.';
  container.replaceChildren(message);
  return () => message.remove();
}
function disposeGroup(group: THREE.Group, sharedTexture: THREE.Texture | null) {
  group.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    if (mesh.material)
      for (const material of Array.isArray(mesh.material)
        ? mesh.material
        : [mesh.material]) {
        const map = (material as THREE.MeshBasicMaterial).map;
        if (map !== sharedTexture) map?.dispose();
        material.dispose();
      }
  });
  group.clear();
}
const decode = (value: string) =>
  Uint8Array.from(atob(value), (character) => character.charCodeAt(0));

export function GeologyBlock({
  volume,
  surfaceImage,
  palette,
  cut,
  verticalScale = 1,
}: {
  volume: Volume;
  surfaceImage: string;
  palette: RockUnit[];
  cut: number;
  verticalScale?: number;
}) {
  const host = useRef<HTMLDivElement>(null),
    state = useRef<SceneState | null>(null);
  const bytes = useMemo(() => decode(volume.data), [volume.data]);
  const heights = useMemo(
    () =>
      volume.surface
        ? new Float32Array(decode(volume.surface.data).buffer)
        : null,
    [volume.surface],
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
    camera.position.set(8.5, 6.5, 9.5);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, -0.5, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 4;
    controls.maxDistance = 26;
    controls.maxPolarAngle = Math.PI * 0.87;
    const block = new THREE.Group();
    scene.add(block);
    const grid = new THREE.GridHelper(18, 18, 0x33424c, 0x25313a);
    grid.position.y = -1.83;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.48;
    scene.add(grid);
    const context: SceneState = {
      renderer,
      camera,
      controls,
      block,
      grid,
      surfaceTexture: null,
      surfaceImage: null,
    };
    state.current = context;
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
      disposeGroup(block, context.surfaceTexture);
      context.surfaceTexture?.dispose();
      context.surfaceTexture = null;
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
      state.current = null;
    };
  }, []);

  useEffect(() => {
    const context = state.current;
    if (!context) return;
    const { block, grid } = context;
    // Rebuilding a cut is synchronous, but decoding its image is not. Keep
    // the loaded surface texture alive across cuts to avoid untextured frames.
    disposeGroup(block, context.surfaceTexture);
    if (!context.surfaceTexture || context.surfaceImage !== surfaceImage) {
      context.surfaceTexture?.dispose();
      const surface = new THREE.TextureLoader().load(surfaceImage, (map) => {
        if (state.current !== context || context.surfaceTexture !== map)
          map.dispose();
      });
      surface.colorSpace = THREE.SRGBColorSpace;
      surface.magFilter = THREE.NearestFilter;
      surface.minFilter = THREE.NearestFilter;
      context.surfaceTexture = surface;
      context.surfaceImage = surfaceImage;
    }
    const [nz, ny, nx] = volume.shape;
    const [xmin, xmax, ymin, ymax, zmin, zmax] = volume.bounds;
    const startY = Math.min(ny - 2, Math.floor(cut * ny));
    const xAt = (i: number) => xmin + ((xmax - xmin) * i) / nx;
    const yAt = (j: number) => ymin + ((ymax - ymin) * j) / ny;
    const heightAt = (i: number, j: number) =>
      heights ? heights[j * (nx + 1) + i] : zmax;
    const ycut = yAt(startY),
      cx = (xmin + xmax) / 2,
      cy = (ymin + ymax) / 2;
    const position = (x: number, y: number, z: number) => [
      x - cx,
      z * verticalScale,
      -(y - cy),
    ];
    grid.position.y = zmin * verticalScale - 0.03;
    const colors = new Uint8Array(256 * 4);
    palette.forEach((unit) => colors.set([...unit.rgb, 255], unit.id * 4));
    const voxel = (x: number, y: number, z: number) =>
      bytes[(z * ny + y) * nx + x];
    // Cut walls end at the exact terrain mesh. Extend the nearest rock label
    // across a partially clipped top voxel to avoid a half-cell display seam.
    const surfaceUnits = new Uint8Array(nx * ny);
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        for (let z = nz - 1; z >= 0; z--) {
          const unit = voxel(x, y, z);
          if (unit) {
            surfaceUnits[y * nx + x] = unit;
            break;
          }
        }
      }
    const wallVoxel = (x: number, y: number, z: number) =>
      voxel(x, y, z) || surfaceUnits[y * nx + x];
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
    function geometry(positions: number[], uvs: number[], indices: number[]) {
      const result = new THREE.BufferGeometry();
      result.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(positions, 3),
      );
      result.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
      result.setIndex(indices);
      result.computeVertexNormals();
      return result;
    }
    function outline(points: number[][]) {
      const segments = points
        .slice(1)
        .flatMap((point, index) => [...points[index], ...point]);
      const line = new THREE.BufferGeometry();
      line.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(segments, 3),
      );
      block.add(
        new THREE.LineSegments(
          line,
          new THREE.LineBasicMaterial({
            color: 0xdce6e8,
            transparent: true,
            opacity: 0.28,
          }),
        ),
      );
    }
    function wall(
      count: number,
      coordinate: (i: number) => [number, number, number],
      map: THREE.Texture,
      shade: number,
    ) {
      const positions: number[] = [],
        uvs: number[] = [],
        indices: number[] = [],
        top: number[][] = [],
        bottom: number[][] = [];
      for (let i = 0; i <= count; i++) {
        const [x, y, height] = coordinate(i),
          low = position(x, y, zmin),
          high = position(x, y, height);
        positions.push(...low, ...high);
        uvs.push(i / count, 0, i / count, (height - zmin) / (zmax - zmin));
        top.push(high);
        bottom.push(low);
        if (i < count) {
          const a = i * 2;
          indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
        }
      }
      block.add(
        new THREE.Mesh(
          geometry(positions, uvs, indices),
          new THREE.MeshBasicMaterial({
            map,
            color: shade,
            side: THREE.DoubleSide,
            alphaTest: 0.1,
          }),
        ),
      );
      outline(top);
      outline(bottom);
      outline([bottom[0], top[0]]);
      outline([bottom[count], top[count]]);
    }
    // The terrain cap shares its edge coordinates with all four cut walls.
    // Its texture is the actual model/DEM intersection used in the score.
    const positions: number[] = [],
      uvs: number[] = [],
      indices: number[] = [];
    for (let j = startY; j <= ny; j++)
      for (let i = 0; i <= nx; i++) {
        positions.push(...position(xAt(i), yAt(j), heightAt(i, j)));
        uvs.push(i / nx, j / ny);
        if (i < nx && j < ny) {
          const a = (j - startY) * (nx + 1) + i,
            b = a + nx + 1;
          indices.push(a, a + 1, b, a + 1, b + 1, b);
        }
      }
    const cap = geometry(positions, uvs, indices),
      normals = cap.getAttribute('normal');
    const sun = new THREE.Vector3(-0.45, 1, 0.65).normalize(),
      shading: number[] = [];
    for (let i = 0; i < normals.count; i++) {
      const dot =
        normals.getX(i) * sun.x +
        normals.getY(i) * sun.y +
        normals.getZ(i) * sun.z;
      const brightness = 0.42 + 0.58 * Math.max(0, dot);
      shading.push(brightness, brightness, brightness);
    }
    cap.setAttribute('color', new THREE.Float32BufferAttribute(shading, 3));
    block.add(
      new THREE.Mesh(
        cap,
        new THREE.MeshBasicMaterial({
          map: context.surfaceTexture,
          vertexColors: true,
          side: THREE.DoubleSide,
          alphaTest: 0.1,
        }),
      ),
    );
    wall(
      nx,
      (i) => [xAt(i), ycut, heightAt(i, startY)],
      texture(nx, nz, (x, z) => wallVoxel(x, startY, z)),
      0xe6ebf0,
    );
    wall(
      nx,
      (i) => [xAt(i), ymax, heightAt(i, ny)],
      texture(nx, nz, (x, z) => wallVoxel(x, ny - 1, z)),
      0xd0d9e2,
    );
    wall(
      ny - startY,
      (j) => [xmin, yAt(j + startY), heightAt(0, j + startY)],
      texture(ny - startY, nz, (y, z) => wallVoxel(0, y + startY, z)),
      0xc7d1db,
    );
    wall(
      ny - startY,
      (j) => [xmax, yAt(j + startY), heightAt(nx, j + startY)],
      texture(ny - startY, nz, (y, z) => wallVoxel(nx - 1, y + startY, z)),
      0xc7d1db,
    );
    block.add(
      new THREE.Mesh(
        geometry(
          [
            ...position(xmin, ycut, zmin),
            ...position(xmax, ycut, zmin),
            ...position(xmax, ymax, zmin),
            ...position(xmin, ymax, zmin),
          ],
          [0, 0, 1, 0, 1, 1, 0, 1],
          [0, 1, 2, 0, 2, 3],
        ),
        new THREE.MeshBasicMaterial({
          map: texture(nx, ny - startY, (x, y) => voxel(x, y + startY, 0)),
          color: 0xc0cad4,
          side: THREE.DoubleSide,
        }),
      ),
    );
  }, [volume, surfaceImage, palette, cut, bytes, heights, verticalScale]);
  return (
    <div
      className="geology-canvas"
      ref={host}
      aria-label="Interactive geology with measured terrain. Drag to orbit and move the cutaway to reveal its interior."
    />
  );
}
