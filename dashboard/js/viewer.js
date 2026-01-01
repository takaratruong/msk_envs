import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { setupLightsSky, setupAxes, setupGround, setupLanes, setupNumbers } from "./scene.js";
import { loadModel, loadCollider } from "./loader.js";
import { drawMuscleCapsule, drawMuscleLine, resetMuscles } from "./muscle.js";

document.addEventListener("DOMContentLoaded", () => {
    THREE.Object3D.DefaultUp = new THREE.Vector3(0, 1, 0);

    const container = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const width = container.clientWidth;
    const height = container.clientHeight;
    let fullScreenMode = false;
    let aspect = (width / 3) / height;

    function updateCameras(com) {
        com[1] = 1.0; // fix for now
        if (autoFollow1) {
            camera1.position.set(com[0], com[1] + 0.5, com[2] + 2.25);
            controls1.target.set(com[0], com[1], com[2]);
        }

        if (autoFollow2) {
            camera2.position.set(com[0] + 2.25, com[1] + 0.5, com[2]);
            controls2.target.set(com[0], com[1], com[2]);
        }

        if (autoFollow3) {
            camera3.position.set(com[0] - 1.5, com[1] + 1.0, com[2]);
            controls3.target.set(com[0], com[1], com[2]);
        }
    }


    // Frame loading
    let frames = [];
    let currentObjects = [];
    let currentFrame = 0;
    function loadFrame(frameIndex) {
        // Clear scene
        currentObjects.forEach(obj => scene.remove(obj));
        currentObjects = [];

        // Load frame
        const frame = frames[frameIndex];
        if (!frame) return;

        // Follow com
        updateCameras(frame.cam_pos)

        // Write time in text. Add directl to the scene
        const time = frame.time;
        timeElement.textContent = "(t = " + time.toFixed(3) + "s)";

        // Draw objects (visual/collider)
        const drawVisuals = drawVisualsCheckbox.checked;
        const drawSphereColliders = drawSphereCollidersCheckbox.checked;
        const drawCapsuleColliders = drawCapsuleCollidersCheckbox.checked;

        if (drawVisuals) {
            for (const obj of frame.visuals) {
                const color = 0xF1ECE4;
                loadModel(obj.mesh_file, obj.opacity, color, object => {
                    object.scale.set(...obj.scale);
                    object.position.set(...obj.pos);
                    object.quaternion.set(obj.rot[1], obj.rot[2], obj.rot[3], obj.rot[0]);
                    scene.add(object);
                    currentObjects.push(object);
                });
            }
        }

        for (const obj of frame.colliders) {
            const color = 0x87CEEB;
            loadCollider(drawSphereColliders, drawCapsuleColliders,
                obj.geom_type, obj.scale, obj.rot, color, object => {
                object.position.set(...obj.pos);
                object.castShadow = true;
                object.receiveShadow = true;
                scene.add(object);
                currentObjects.push(object);
            });
        }

        // Draw muscles
        const drawMuscles = drawMusclesCheckbox.checked;
        if (drawMuscles && frame.muscles) {
            // add a point for
            for (const muscle of frame.muscles) {
                // drawMuscleCapsule(muscle, object => {
                drawMuscleLine(muscle, object => {
                    scene.add(object);
                    currentObjects.push(object);
                });
            }
        }

        currentFrame = frameIndex;
    }

    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    const camera1 = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    const camera2 = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    const camera3 = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
    camera1.up.set(0, 1, 0);
    camera2.up.set(0, 1, 0);
    camera3.up.set(0, 1, 0);
    const controls1 = new OrbitControls(camera1, renderer.domElement);
    const controls2 = new OrbitControls(camera2, renderer.domElement);
    const controls3 = new OrbitControls(camera3, renderer.domElement);

    const drawVisualsCheckbox = document.getElementById("drawVisuals");
    const drawSphereCollidersCheckbox = document.getElementById("drawSphereColliders");
    const drawCapsuleCollidersCheckbox = document.getElementById("drawCapsuleColliders");
    const drawMusclesCheckbox = document.getElementById("drawMuscles");
    const timeElement = document.getElementById("timeValue");

    const fullScreen = document.getElementById("fullScreenButton");

    // Auto follow camera
    const resetView1 = document.getElementById("resetButton1");
    const resetView2 = document.getElementById("resetButton2");
    const resetView3 = document.getElementById("resetButton3");
    resetView1.style.display = "none";
    resetView2.style.display = "none";
    resetView3.style.display = "none";
    let autoFollow1 = true;
    let autoFollow2 = true;
    let autoFollow3 = true;

    function onMouseMove(event) {
        if (fullScreenMode) {
            controls1.enabled = true;
            controls2.enabled = false;
            controls3.enabled = false;
            return;
        }
        const x = event.clientX;
        if (x < container.getBoundingClientRect().left + width / 3) {
            controls1.enabled = true;
            controls2.enabled = false;
            controls3.enabled = false;
        } else if (x < container.getBoundingClientRect().left + 2 * width / 3) {
            controls1.enabled = false;
            controls2.enabled = true;
            controls3.enabled = false;
        } else {
            controls1.enabled = false;
            controls2.enabled = false;
            controls3.enabled = true;
        }
    }

    // Event listeners
    document.addEventListener("mousemove", onMouseMove, false);
    window.addEventListener("sliderChanged", function(event) {
        const frameIndex = parseInt(event.detail);
        loadFrame(frameIndex);
    });

    window.addEventListener("loadTrajectory", function(event) {
        frames = event.detail;
        loadFrame(0);
    });

    window.addEventListener("resetViewer", function(event) {
        frames = [];
        currentObjects.forEach(obj => scene.remove(obj));
        currentObjects = [];
        currentFrame = 0;
        resetMuscles();
    });

    fullScreen.addEventListener('click', () => {
        fullScreen.textContent = fullScreenMode ? "Full Screen" : "Panel View";
        fullScreenMode = !fullScreenMode;
        // update camera aspect
        if (fullScreenMode) {
            camera1.aspect = width / height;
            camera1.updateProjectionMatrix();
        } else {
            camera1.aspect = (width / 3) / height;
            camera2.aspect = (width / 3) / height;
            camera3.aspect = (width / 3) / height;
            camera1.updateProjectionMatrix();
            camera2.updateProjectionMatrix();
            camera3.updateProjectionMatrix();
        }
        loadFrame(currentFrame);
    });

    controls1.addEventListener("start", () => {
        resetView1.style.display = "block";
        autoFollow1 = false;
    });
    resetView1.addEventListener('click', () => {
        resetView1.style.display = "none";
        autoFollow1 = true;
        loadFrame(currentFrame);
    });
    controls2.addEventListener("start", () => {
        resetView2.style.display = "block";
        autoFollow2 = false;
    });
    resetView2.addEventListener('click', () => {
        resetView2.style.display = "none";
        autoFollow2 = true;
        loadFrame(currentFrame);
    });
    controls3.addEventListener("start", () => {
        resetView3.style.display = "block";
        autoFollow3 = false;
    });
    resetView3.addEventListener('click', () => {
        resetView3.style.display = "none";
        autoFollow3 = true;
        loadFrame(currentFrame);
    });


    drawVisualsCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawSphereCollidersCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawCapsuleCollidersCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });
    drawMusclesCheckbox.addEventListener("change", () => {
        loadFrame(currentFrame);
    });


    function animate() {
        controls1.update();
        controls2.update();
        controls3.update();

        const borderWidth = 2;
        const viewWidth = (width - borderWidth) / 3;
        const viewHeight = height;

        if (fullScreenMode) {
            renderer.setViewport(0, 0, width, height);
            renderer.setScissor(0, 0, width, height);
            renderer.render(scene, camera1);
            return;
        }

        renderer.setScissorTest(true);

        renderer.setViewport(0, 0, viewWidth, viewHeight);
        renderer.setScissor(0, 0, viewWidth, viewHeight);
        renderer.render(scene, camera1);

        renderer.setViewport(viewWidth + borderWidth, 0, viewWidth, viewHeight);
        renderer.setScissor(viewWidth + borderWidth, 0, viewWidth, viewHeight);
        renderer.render(scene, camera2);

        renderer.setViewport(2 * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
        renderer.setScissor(2 * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
        renderer.render(scene, camera3);


        renderer.setScissorTest(false);
    }

    setupGround(scene, new THREE.Vector3(0, 0, 0));
    setupGround(scene, new THREE.Vector3(50, 0, 0));
    setupNumbers(scene);
    setupLightsSky(scene);
    setupLanes(scene);
    setupAxes(scene);
    renderer.setAnimationLoop(animate);
});