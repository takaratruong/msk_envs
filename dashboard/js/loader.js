import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { VTKLoader } from 'three/addons/loaders/VTKLoader.js';

const objLoader = new OBJLoader();
const vtpLoader = new VTKLoader();


const modelCache = {};

function loadModel(file, opacity, color, callback) {
    opacity = opacity !== undefined ? opacity : 1.0;
    const cache_key = file + color + opacity;

    if (modelCache[cache_key]) {
        callback(modelCache[cache_key].clone());
        return;
    }

    const isVtp = file.toLowerCase().endsWith('.vtp');
    if (isVtp) {
        vtpLoader.load(file, geometry => {
            geometry.computeVertexNormals();

            const material = new THREE.MeshStandardMaterial({
                metalness: 0.3,
                roughness: 0.7,
                flatShading: false,
                color: opacity < 1.0 ? 0x00ffff : color,
                transparent: opacity < 1.0,
                opacity: opacity,
            });

            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;

            modelCache[cache_key] = mesh;
            callback(mesh.clone());
        });
    } else {
            objLoader.load(file, obj => {
            obj.traverse(child => {
                if (!child.isMesh) return;
                child.castShadow = true;
                child.receiveShadow = true;
                child.material = new THREE.MeshStandardMaterial({
                    metalness: 0.3,
                    roughness: 0.7,
                    flatShading: false,
                    color: opacity < 1.0 ? 0x00ffff : color,
                    transparent: opacity < 1.0,
                    opacity: opacity,
                });
            });
            modelCache[cache_key] = obj;
            callback(obj.clone());
        });
    }
}


function loadCollider (spheres, capsules, geomType, scale, rot, color, callback) {
    const opacity = 0.8;
    if (geomType === 0) {
        // Plane
    } else if (geomType === 2 && spheres) {
        // Create a sphere geometry
        const radius = scale[0];
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(radius, 16, 16),
            new THREE.MeshStandardMaterial({ color: color, wireframe: false, transparent: true, opacity: opacity })
        );
        sphere.quaternion.set(rot[1], rot[2], rot[3], rot[0]);
        callback(sphere);
    } else if (geomType === 3 && capsules) {
        // Capsule
        const radius = scale[0];
        const half_height = scale[1];
        const capsule = new THREE.Mesh(
            new THREE.CapsuleGeometry(radius, 2 * half_height, 8, 16),
            new THREE.MeshStandardMaterial({ color: color, wireframe: false, transparent: true, opacity: opacity })
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