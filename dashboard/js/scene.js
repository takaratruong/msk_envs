import * as THREE from 'three';
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
const textureLoader = new THREE.TextureLoader();
const planeTexture = textureLoader.load('assets/textures/plane.png');


function setupLightsSky(scene) {
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.);
    const dirLight = new THREE.DirectionalLight(0xffffff, 2);
    dirLight.position.set(50, 50, 50); // Changed Y from -50 to 50
    dirLight.castShadow = true;
    dirLight.shadow.camera.left = -10;
    dirLight.shadow.camera.right = 100;
    dirLight.shadow.camera.top = 50;
    dirLight.shadow.camera.bottom = -50;
    dirLight.shadow.camera.near = 1;
    dirLight.shadow.camera.far = 200;

    dirLight.shadow.mapSize.width = 16384;
    dirLight.shadow.mapSize.height = 16384;
    dirLight.shadow.bias = -0.0001;

    scene.add(ambientLight);
    scene.add(dirLight);

    // Skybox (simple blue cube)
    const size = 500;
    const skybox = new THREE.Mesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshBasicMaterial({color: 0x87ceeb, side: THREE.BackSide})
    );
    scene.add(skybox);

    // Add xyz axes helper, thick
    const axesHelper = new THREE.AxesHelper(1);
    axesHelper.material.linewidth = 16;
    scene.add(axesHelper);
}

function setupAxes(scene) {
    const axesHelper = new THREE.Group();
    const axisLength = 1;
    const axisRadius = 0.025;

    function createAxis(color, rotationAxis, rotationAngle) {
        const geometry = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength, 12);
        const material = new THREE.MeshBasicMaterial({ color });
        const axis = new THREE.Mesh(geometry, material);
        axis.position.y = axisLength / 2;
        if (rotationAxis) {
            axis.rotateOnAxis(rotationAxis, rotationAngle);
            axis.position.set(0, 0, 0);
            axis.translateY(axisLength / 2);
        }
        return axis;
    }

    // xyz
    axesHelper.add(createAxis(0x646efa, new THREE.Vector3(0, 0, 1), -Math.PI / 2));
    axesHelper.add(createAxis(0xef553b));
    axesHelper.add(createAxis(0x00cc96, new THREE.Vector3(1, 0, 0), Math.PI / 2));
    scene.add(axesHelper);
}

function setupGround(scene, position) {
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({
        map: planeTexture,
        metalness: 0.3,
        roughness: 0.7,
        side: THREE.DoubleSide
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.position.set(position.x, position.y, position.z);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
}

function setupLanes(scene) {
    // Lane 1
    const laneWidth = 100;     // Length of the lane
    const laneThickness = 0.05; // Thickness of the line (height of the plane)

    const laneGeometry1 = new THREE.PlaneGeometry(laneWidth, laneThickness);
    const laneMaterial1 = new THREE.MeshBasicMaterial({ color: 0xFFFFFF, side: THREE.DoubleSide });

    const lane1 = new THREE.Mesh(laneGeometry1, laneMaterial1);
    lane1.position.set(50, 0.01, -0.7);
    lane1.rotation.x = -Math.PI / 2; // Rotate to lay flat on the XZ plane

    scene.add(lane1);

    // Lane 2
    const laneGeometry2 = new THREE.PlaneGeometry(laneWidth, laneThickness);
    const laneMaterial2 = new THREE.MeshBasicMaterial({ color: 0xFFFFFF, side: THREE.DoubleSide });

    const lane2 = new THREE.Mesh(laneGeometry2, laneMaterial2);
    lane2.position.set(50, 0.01, 0.7);
    lane2.rotation.x = -Math.PI / 2;

    scene.add(lane2);
}

function setupNumbers(scene) {
    const loader = new FontLoader();
    loader.load("assets/fonts/helvetiker.json", (font) => {

        const step = 10;          // Distance between numbers
        const max = 100;          // Numbers from 0 → 100
        const size = 1;           // Character size

        for (let x = 0; x <= max; x += step) {
            const textGeo = new TextGeometry(String(x), {
                font: font,
                size: size,
                depth: 0.02,      // Use 'depth' instead of 'height'
                curveSegments: 16,
            });

            textGeo.computeBoundingBox();
            const textMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
            const text = new THREE.Mesh(textGeo, textMat);

            text.position.set(x, 0.05, -1.5);

            text.castShadow = true;
            text.receiveShadow = true;

            scene.add(text);
        }
    })
}



export { setupLightsSky, setupAxes, setupGround, setupLanes, setupNumbers };