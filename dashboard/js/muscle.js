import * as THREE from 'three';

const muscleCache = {};

function activationToColor(activation) {
    // 0 is blue. 1 is red.
    const r = activation;
    const g = 0.0;
    const b = 1.0 - activation;
    return new THREE.Color().setRGB(r, g, b);
}

function resetMuscles() {
    for (const key in muscleCache) {
        const cached = muscleCache[key];
        if (Array.isArray(cached)) {
            // Capsule cache
            const [tendons, muscles] = cached;
            tendons.forEach(tendon => {
                tendon.geometry.dispose();
                tendon.material.dispose();
            });
            muscles.forEach(muscle => {
                muscle.geometry.dispose();
                muscle.material.dispose();
            });
        } else {
            // Line2 cache
            cached.geometry.dispose();
            cached.material.dispose();
        }
    }
}

function updateCapsule(capsule, p1, p2, radius, color) {
    const direction = new THREE.Vector3().subVectors(p2, p1);
    const length = direction.length();
    const dir = direction.normalize();
    const currentHeight = capsule.geometry.parameters.height;
    if (Math.abs(currentHeight - length) > 0.001) {
        capsule.geometry.dispose();
        capsule.geometry = new THREE.CapsuleGeometry(radius, length, 8, 16);
    }
    const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    capsule.position.copy(midpoint);
    capsule.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

    // Update the material color for this specific capsule
    capsule.material.color.copy(color);
    capsule.material.needsUpdate = true;
}

function drawMuscleCapsule(muscle, addObjectCallback) {
    const tendonRadius = Math.sqrt(muscle.max_isometric_force) / 8000;
    const muscleRadius = Math.sqrt(muscle.max_isometric_force) / 4000;
    const color = activationToColor(muscle.activation);

    // Check cache
    let currTendons = [];
    let currMuscles = [];
    if (muscleCache[muscle.name]) {
        currTendons = muscleCache[muscle.name][0];
        currMuscles = muscleCache[muscle.name][1];
    } else {
        // Create new tendons and muscles with individual materials
        for (let i = 0; i < muscle.points.length - 1; i++) {
            // Create unique materials for each capsule
            const tendonMaterial = new THREE.MeshStandardMaterial({
                color: 0xFFFFC5,
                side: THREE.DoubleSide,
                flatShading: false,
                metalness: 0.0,
                roughness: 1.0
            });

            const muscleMaterial = new THREE.MeshStandardMaterial({
                color: 0xFF8888,
                side: THREE.DoubleSide,
                flatShading: false,
                metalness: 0.0,
                roughness: 1.0
            });

            const tendonGeometry = new THREE.CapsuleGeometry(tendonRadius, 0, 8, 16);
            const tendon = new THREE.Mesh(tendonGeometry, tendonMaterial);
            const muscleGeometry = new THREE.CapsuleGeometry(muscleRadius, 0, 8, 16);
            const muscleBody = new THREE.Mesh(muscleGeometry, muscleMaterial);

            // shadows are nice
            tendon.castShadow = true;
            tendon.receiveShadow = true;
            muscleBody.castShadow = true;
            muscleBody.receiveShadow = true;

            currTendons.push(tendon);
            currMuscles.push(muscleBody);
        }
        muscleCache[muscle.name] = [currTendons, currMuscles];
    }

    // Update tendons
    for (let i = 0; i < muscle.points.length - 1; i++) {
        const p1 = new THREE.Vector3(...muscle.points[i]);
        const p2 = new THREE.Vector3(...muscle.points[i + 1]);
        const capsule = currTendons[i];
        updateCapsule(capsule, p1, p2, tendonRadius, new THREE.Color(0xFFFFC5));
        addObjectCallback(capsule);
    }

    // Update muscles
    const muscleLength = muscle.path_length - muscle.tendon_length;
    const half_tendon_length = muscle.tendon_length / 2;

    // Hide all muscles to start
    for (let i = 0; i < currMuscles.length; i++) {
        currMuscles[i].visible = false;
    }
    if (muscleLength <= 0) return;

    // Figure out where the muscle starts (should have half tendon length before and after)
    let accumulatedLength = 0;
    let muscleStartPoint = new THREE.Vector3(...muscle.points[0]);
    let muscleStartIndex = 0;
    for (let i = 0; i < muscle.points.length - 1; i++) {
        const p1 = new THREE.Vector3(...muscle.points[i]);
        const p2 = new THREE.Vector3(...muscle.points[i + 1]);

        // check if exceed half tendon length
        const segment = new THREE.Vector3().subVectors(p2, p1);
        const segmentLength = segment.length();
        if (accumulatedLength + segmentLength > half_tendon_length) {
            const rem = half_tendon_length - accumulatedLength;
            muscleStartPoint.add(segment.setLength(rem));
            muscleStartIndex = i;
            break;
        }

        accumulatedLength += segmentLength;
        muscleStartPoint.copy(p2);
    }

    // Draw muscles with activation color
    accumulatedLength = 0;
    for (let i = muscleStartIndex; i < muscle.points.length - 1; i++) {
        const p1 = new THREE.Vector3(...muscle.points[i]);
        const p2 = new THREE.Vector3(...muscle.points[i + 1]);
        if (i === muscleStartIndex) p1.copy(muscleStartPoint);

        const segment = new THREE.Vector3().subVectors(p2, p1);
        const segmentLength = segment.length();

        // Full vs partial segment
        if (accumulatedLength + segmentLength < muscleLength) {
            const muscleBody = currMuscles[i];
            updateCapsule(muscleBody, p1, p2, muscleRadius, color);
            muscleBody.visible = true;
            addObjectCallback(muscleBody);
            accumulatedLength += segmentLength;
        } else {
            const rem = muscleLength - accumulatedLength;
            const end = new THREE.Vector3().addVectors(p1, segment.setLength(rem));
            const muscleBody = currMuscles[i];
            updateCapsule(muscleBody, p1, end, muscleRadius, color);
            muscleBody.visible = true;
            addObjectCallback(muscleBody);
            break;
        }
    }
}

function drawMuscleLine(muscle, addObjectCallback) {
    const color = activationToColor(muscle.activation);
    const muscleRadius = 0.005;

    let capsules = [];
    if (muscleCache[muscle.name]) {
        capsules = muscleCache[muscle.name];
    } else {
        // Create capsules for each segment
        for (let i = 0; i < muscle.points.length - 1; i++) {
            const muscleMaterial = new THREE.MeshStandardMaterial({
                color: 0xFF8888,
                side: THREE.DoubleSide,
                flatShading: false,
                metalness: 0.0,
                roughness: 1.0
            });

            const muscleGeometry = new THREE.CapsuleGeometry(muscleRadius, 0, 8, 16);
            const capsule = new THREE.Mesh(muscleGeometry, muscleMaterial);

            capsule.castShadow = true;
            capsule.receiveShadow = true;

            capsules.push(capsule);
        }
        muscleCache[muscle.name] = capsules;
    }

    // Update all capsules
    for (let i = 0; i < muscle.points.length - 1; i++) {
        const p1 = new THREE.Vector3(...muscle.points[i]);
        const p2 = new THREE.Vector3(...muscle.points[i + 1]);
        const capsule = capsules[i];
        updateCapsule(capsule, p1, p2, muscleRadius, color);
        addObjectCallback(capsule);
    }
}

export {drawMuscleCapsule, drawMuscleLine, resetMuscles};