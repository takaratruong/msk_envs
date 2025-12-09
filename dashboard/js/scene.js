import * as THREE from 'three';
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

function setupGround(scene) {
    const groundGeometry = new THREE.PlaneGeometry(100, 100);
    const groundMaterial = new THREE.MeshStandardMaterial({
        map: planeTexture,
        metalness: 0.3,
        roughness: 0.7,
        side: THREE.DoubleSide
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
}

export { setupLightsSky, setupAxes, setupGround };