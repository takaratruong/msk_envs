import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {setupLightsSky, setupAxes, setupGround, setupLanes, setupNumbers} from "./scene.js";
import {loadModel, loadCollider} from "./loader.js";
import {drawMuscleLine, resetMuscles} from "./muscle.js";

const ViewMode = Object.freeze({
    FULLSCREEN: {key: "FULLSCREEN", label: "Full Screen"},
    PANEL: {key: "PANEL", label: "Panel View"},
    PANEL_FEET: {key: "PANEL_FEET", label: "Panel with Feet"},
});


document.addEventListener("DOMContentLoaded", () => {
    THREE.Object3D.DefaultUp = new THREE.Vector3(0, 1, 0);

    const container = document.getElementById("viewer");
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    const width = container.clientWidth;
    const height = container.clientHeight;
    let cameraViewMode = ViewMode.FULLSCREEN;
    let aspect = (width / 3) / height;

    function updateCameras(com, footL, footR) {
        // backwards compatibility
        if (footL === undefined) footL = [com[0], com[1] - 1.0, com[2]];
        if (footR === undefined) footR = [com[0], com[1] - 1.0, com[2]];

        com[1] = 1.0; // fix for now
        if (autoFollow1) {
            cameras[0].position.set(com[0], com[1] + 0.5, com[2] + 2.25);
            controls[0].target.set(com[0], com[1], com[2]);
        }
        if (cameraViewMode === ViewMode.PANEL) {
            if (autoFollow2) {
                cameras[1].position.set(com[0] + 2.25, com[1] + 0.5, com[2]);
                controls[1].target.set(com[0], com[1], com[2]);
            }
            if (autoFollow3) {
                cameras[2].position.set(com[0] - 1.5, com[1] + 1.0, com[2]);
                controls[2].target.set(com[0], com[1], com[2]);
            }
        }
        else if (cameraViewMode === ViewMode.PANEL_FEET) {
            cameras[1].position.set(footR[0], footR[1], footR[2] + 0.5);
            controls[1].target.set(footR[0], footR[1], footR[2]);
            cameras[3].position.set(footR[0] + 0.5, footR[1], footR[2]);
            controls[3].target.set(footR[0], footR[1], footR[2]);

            cameras[2].position.set(footL[0], footL[1], footL[2] - 0.5);
            controls[2].target.set(footL[0], footL[1], footL[2]);
            cameras[4].position.set(footL[0] + 0.5, footL[1], footL[2]);
            controls[4].target.set(footL[0], footL[1], footL[2]);
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
        updateCameras(frame.cam_pos, frame.foot_l_pos, frame.foot_r_pos);

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
            let color = 0x87CEEB;
            if (obj.contact_force) {
                let inContact = obj.contact_force > 0.0;
                color = inContact ? 0x6AEB9D : 0x87CEEB;
            }
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
                drawMuscleLine(muscle.name, muscle, object => {
                    scene.add(object);
                    currentObjects.push(object);
                });
            }
        }

        for (const arrow of frame.arrows) {
            const origin = new THREE.Vector3(arrow.start[0], arrow.start[1], arrow.start[2]);
            const dir = new THREE.Vector3(arrow.direction[0], arrow.direction[1], arrow.direction[2]);
            const length = dir.length();
            if (length < 1e-12) continue;

            dir.normalize();

            // Create cylinder with unit length and thickness based on original length
            const radius = 0.001 * Math.sqrt(length);
            const cylinderLength = 0.2;
            const geometry = new THREE.CylinderGeometry(radius, radius, cylinderLength, 8);
            const material = new THREE.MeshStandardMaterial({color: 0xff0000});
            const cylinder = new THREE.Mesh(geometry, material);

            // Position cylinder at the midpoint
            const midpoint = origin.clone().add(dir.clone().multiplyScalar(cylinderLength / 2));
            cylinder.position.copy(midpoint);
            cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            currentObjects.push(cylinder);
            scene.add(cylinder);

            // Create arrowhead (cone)
            const coneHeight = radius * 4;
            const coneRadius = radius * 2;
            const coneGeometry = new THREE.ConeGeometry(coneRadius, coneHeight, 8);
            const cone = new THREE.Mesh(coneGeometry, material);

            // Position cone at the end of the cylinder
            const conePosition = origin.clone().add(dir.clone().multiplyScalar(cylinderLength + coneHeight / 2));
            cone.position.copy(conePosition);
            cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            currentObjects.push(cone);
            scene.add(cone);
        }

        currentFrame = frameIndex;
    }

    const renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Panel cameras
    let cameras = [];
    let controls = [];
    for (let i = 0; i < 5; i++) {
        const cam = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        const control = new OrbitControls(cam, renderer.domElement);
        cam.up.set(0, 1, 0);

        cameras.push(cam);
        controls.push(control);
    }
    cameras[0].aspect = width / height;
    cameras[0].updateProjectionMatrix();

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
        if (cameraViewMode === ViewMode.FULLSCREEN) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 0);
            }
            return;
        }
        const x = event.clientX;
        if (x < container.getBoundingClientRect().left + width / 3) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 0);
            }
        } else if (x < container.getBoundingClientRect().left + 2 * width / 3) {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 1);
            }
        } else {
            for (let i = 0; i < controls.length; i++) {
                controls[i].enabled = (i === 2);
            }
        }
    }

    // Event listeners
    document.addEventListener("mousemove", onMouseMove, false);
    window.addEventListener("sliderChanged", function (event) {
        const frameIndex = parseInt(event.detail);
        loadFrame(frameIndex);
    });

    window.addEventListener("loadTrajectory", function (event) {
        frames = event.detail;
        loadFrame(0);
    });

    window.addEventListener("resetViewer", function (event) {
        frames = [];
        currentObjects.forEach(obj => scene.remove(obj));
        currentObjects = [];
        currentFrame = 0;
        resetMuscles();
    });

    fullScreen.addEventListener('click', () => {
        let viewModes = Object.values(ViewMode);

        let currentIndex = viewModes.indexOf(cameraViewMode);
        let nextIndex = (currentIndex + 1) % viewModes.length;
        let nextNextIndex = (currentIndex + 2) % viewModes.length;

        cameraViewMode = viewModes[nextIndex];
        let nextMode = viewModes[nextNextIndex];
        fullScreen.textContent = String(nextMode.label);
        // update camera aspect
        if (cameraViewMode === ViewMode.FULLSCREEN) {
            cameras[0].aspect = width / height;
            cameras[0].updateProjectionMatrix();
        } else if (cameraViewMode === ViewMode.PANEL) {
            for (let i = 0; i < cameras.length; i++) {
                cameras[i].aspect = (width / 3) / height;
                cameras[i].updateProjectionMatrix();
            }
        } else if (cameraViewMode === ViewMode.PANEL_FEET) {
            cameras[0].aspect = (width / 3) / height;
            cameras[0].updateProjectionMatrix();
            // Split the other 2/3 among the remaining four cameras
            for (let i = 1; i < cameras.length; i++) {
                cameras[i].aspect = (width / 3) / (height / 2);
                cameras[i].updateProjectionMatrix();
            }
        }
        loadFrame(currentFrame);
    });

    controls[0].addEventListener("start", () => {
        resetView1.style.display = "block";
        autoFollow1 = false;
    });
    resetView1.addEventListener('click', () => {
        resetView1.style.display = "none";
        autoFollow1 = true;
        loadFrame(currentFrame);
    });
    controls[1].addEventListener("start", () => {
        resetView2.style.display = "block";
        autoFollow2 = false;
    });
    resetView2.addEventListener('click', () => {
        resetView2.style.display = "none";
        autoFollow2 = true;
        loadFrame(currentFrame);
    });
    controls[2].addEventListener("start", () => {
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
        for (let i = 0; i < controls.length; i++) {
            controls[i].update();
        }

        const borderWidth = 2;
        const viewWidth = (width - borderWidth) / 3;
        const viewHeight = height;

        if (cameraViewMode === ViewMode.FULLSCREEN) {
            renderer.setViewport(0, 0, width, height);
            renderer.setScissor(0, 0, width, height);
            renderer.render(scene, cameras[0]);
            return;
        } else if (cameraViewMode === ViewMode.PANEL) {
            renderer.setScissorTest(true);
            for (let i = 0; i < 3; i++) {
                renderer.setViewport(i * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
                renderer.setScissor(i * (viewWidth + borderWidth), 0, viewWidth, viewHeight);
                renderer.render(scene, cameras[i]);
            }
            renderer.setScissorTest(false);
        } else if (cameraViewMode === ViewMode.PANEL_FEET) {
            renderer.setScissorTest(true);

            // Main camera (left third)
            renderer.setViewport(0, 0, viewWidth, viewHeight);
            renderer.setScissor(0, 0, viewWidth, viewHeight);
            renderer.render(scene, cameras[0]);

            // Right cameras (right two thirds, split each third vertically among two cameras each)
            const halfViewHeight = viewHeight / 2;
            for (let i = 1; i < cameras.length; i++) {
                const col = (i - 1) % 2;
                const row = 1 - Math.floor((i - 1) / 2);
                const x = viewWidth + borderWidth + col * (viewWidth + borderWidth);
                const y = row * (halfViewHeight + borderWidth);

                renderer.setViewport(x, y, viewWidth, halfViewHeight);
                renderer.setScissor(x, y, viewWidth, halfViewHeight);
                renderer.render(scene, cameras[i]);
            }
            renderer.setScissorTest(false);

        }
    }

    setupGround(scene, new THREE.Vector3(-50, 0, 0));
    setupGround(scene, new THREE.Vector3(50, 0, 0));
    setupNumbers(scene);
    setupLightsSky(scene);
    setupLanes(scene);
    setupAxes(scene);
    renderer.setAnimationLoop(animate);
});