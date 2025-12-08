import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { LoopSubdivision } from 'https://unpkg.com/three-subdivide/build/index.module.js';

const objLoader = new OBJLoader();
const textureLoader = new THREE.TextureLoader();
const planeTexture = textureLoader.load('assets/textures/plane.png');

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

                if (objFile.includes('plane')) {
                    child.material = new THREE.MeshStandardMaterial({
                        map: planeTexture,
                        metalness: 0.3,
                        roughness: 0.7,
                    });
                } else {
                    child.material = new THREE.MeshStandardMaterial({
                        metalness: 0.3,
                        roughness: 0.7,
                        flatShading: false,
                        color: color
                    });
                }

                if (!objFile.includes('plane') && !objFile.includes('cube')) {
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
                }
            });
            modelCache[objFile + color] = obj;
            callback(obj.clone());
        });
    }
}

export { loadModel };