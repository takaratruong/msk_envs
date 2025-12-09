import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { LoopSubdivision } from 'https://unpkg.com/three-subdivide/build/index.module.js';

const objLoader = new OBJLoader();

const modelCache = {};

function loadModel(objFile, color, callback) {
    if (modelCache[objFile + color]) {
        callback(modelCache[objFile + color].clone());
    } else {
        objLoader.load(objFile, obj => {
            obj.traverse(child => {
                if (!child.isMesh) return;
                child.castShadow = true;
                child.receiveShadow = true;
                child.material = new THREE.MeshStandardMaterial({
                    metalness: 0.3,
                    roughness: 0.7,
                    flatShading: false,
                    color: color
                });

                // Smoothen
                const iterations = 0;
                const params = {
                    split: false,
                    uvSmooth: true,
                    preserveEdges: false,
                    flatOnly: false,
                    maxTriangles: 5000,
                };

                const geometry = LoopSubdivision.modify(child.geometry, iterations, params);
                child.geometry.dispose();
                child.geometry = geometry;
            });
            modelCache[objFile + color] = obj;
            callback(obj.clone());
        });
    }
}

function loadCollider (geomType, scale, rot, color, callback) {
    if (geomType === 0) {
        // Plane
    } else if (geomType === 2) {
        // Create a sphere geometry
        const radius = scale[0];
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(radius, 16, 16),
            new THREE.MeshStandardMaterial({ color: color, wireframe: false})
        );
        sphere.quaternion.set(rot[1], rot[2], rot[3], rot[0]);
        callback(sphere);
    } else if (geomType === 3) {
        // Capsule
        const radius = scale[0];
        const half_height = scale[1];
        const capsule = new THREE.Mesh(
            new THREE.CapsuleGeometry(radius, 2 * half_height, 8, 16),
            new THREE.MeshStandardMaterial({ color: color, wireframe: false})
        );
        // Create quaternion for Z-up rotation (90 degrees around X-axis)
        const zUpQuat = new THREE.Quaternion();
        zUpQuat.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

        // Create quaternion from rotation data
        const dataQuat = new THREE.Quaternion(rot[1], rot[2], rot[3], rot[0]);

        // Multiply quaternions: first apply Z-up, then apply data rotation
        capsule.quaternion.copy(dataQuat).multiply(zUpQuat);

        callback(capsule);
    }
}

export { loadModel, loadCollider };