import { AmbientLight, AnimationClip, AnimationMixer, Bone, BoxGeometry, Clock, Color, CylinderGeometry, GridHelper, Group, KeyframeTrack, LineBasicMaterial, Matrix3, Matrix4, Mesh, MeshStandardMaterial, Object3D, PerspectiveCamera, Plane, PlaneGeometry, PlaneHelper, Ray, Raycaster, Scene, Skeleton, SkeletonHelper, SkinnedMesh, Sphere, SphereGeometry, Vector2, Vector3, VectorKeyframeTrack, WebGLRenderer } from 'three';  
import Controls from './controls';

export default class MotionEditor {
    renderer: WebGLRenderer;
    skeletonHelper: SkeletonHelper;
    scene: Scene;
    mixer: AnimationMixer;
    camera: PerspectiveCamera;
    clock: Clock;
    id: number;
    sphereMeshes: Mesh[];
    cylinderMeshes: Mesh[];
    track: number[][][];
    frame: number;
    raycaster: Raycaster;
    pointer: Vector2;
    hovered: Object3D;
    clicked: Object3D;
    clickedIndex: number;
    controls: Controls;

    constructor(canvas: HTMLCanvasElement, motionSequence: number[][][], controls: Controls) {
        this.init(canvas);

        this.controls = controls;
        this.controls.hide();

        const [skeleton, track] = this.constructSkeleton(motionSequence);

        this.track = track;

        this.frame = 0;

        this.skeletonHelper = new SkeletonHelper(skeleton.bones[0]);
        // @ts-ignore
        this.skeletonHelper.skeleton = skeleton

        if (this.skeletonHelper.material instanceof LineBasicMaterial) {
            this.skeletonHelper.material.linewidth = 10
        }

        // this.scene.add(this.skeletonHelper);

        const boneContainer = new Group();
        boneContainer.add(skeleton.bones[0]);
        this.scene.add(boneContainer);

        this.sphereMeshes = [];
        this.cylinderMeshes = [];

        const cylinderMaterial = new MeshStandardMaterial();
        cylinderMaterial.color.setRGB(0, 0, 255);

        skeleton.bones.forEach((bone, index) => {
            const sphereGeometry = new SphereGeometry(3.2);

            const sphereMaterial = new MeshStandardMaterial();
            sphereMaterial.color.setRGB(255, 0, 0);

            const sphereMesh = new Mesh(sphereGeometry, sphereMaterial);
            setSphereMesh(sphereMesh, bone);

            this.sphereMeshes.push(sphereMesh);
            this.scene.add(sphereMesh);
        });

        skeleton.bones.forEach((bone, index) => {
            if (!(bone.parent instanceof Bone)) return;

            const height = bone.parent.position.distanceTo(bone.position);

            const cylinderGeometry = new CylinderGeometry(1.5, 1.5, height);

            const cylinderMesh = new Mesh(cylinderGeometry, cylinderMaterial);
            setCylinderMesh(cylinderMesh, bone);

            this.cylinderMeshes.push(cylinderMesh);

            this.scene.add(cylinderMesh);
        });

        this.initControls();

        this.animate();
        
        // this.mixer = new AnimationMixer(this.skeletonHelper);
        // this.mixer.clipAction(animationClip).play();

    }

    init(canvas: HTMLCanvasElement) {
        this.camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 10000);
        this.camera.position.set(200, 100, 100);
        this.camera.lookAt(0, 0, 0);

        this.scene = new Scene();
        this.scene.background = new Color(0xeeeeee);

        this.scene.add(new AmbientLight());

        const ground = new Mesh(
            new PlaneGeometry(300, 300),
            new MeshStandardMaterial({ color: 0x333333 })
        );
        ground.rotation.set(-Math.PI / 2, 0, 0);
        ground.position.set(0, -100, 0);
        this.scene.add(ground);


        this.renderer = new WebGLRenderer({ antialias: true, canvas: canvas });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    resizeCanvasToDisplaySize() {
        const canvas = this.renderer.domElement;
        // look up the size the canvas is being displayed
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;

        // adjust displayBuffer size to match
        if (canvas.width !== width || canvas.height !== height) {
            // you must pass false here or three.js sadly fights the browser
            this.renderer.setSize(width, height, false);
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
        }
    }

    initControls() {
        this.raycaster = new Raycaster();
        this.pointer = new Vector2();

        window.addEventListener('keydown', (event: KeyboardEvent) => {
            switch (event.key) {
                case 'ArrowLeft':
                    this.frame = (this.frame + this.track.length - 30) % this.track.length;
                    break;
                case 'ArrowRight':
                    this.frame = (this.frame + 30) % this.track.length;
                    break;
            }
        });

        window.addEventListener('pointermove', (event: MouseEvent) => {
            this.pointer.x = ( event.clientX / window.innerWidth ) * 2 - 1;
	        this.pointer.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
        });

        window.addEventListener('click', () => {
            this.handleClick();
        })
    }

    handleIntersection() {
        this.raycaster.setFromCamera(this.pointer, this.camera);

        const intersects = this.raycaster.intersectObjects(this.sphereMeshes);

        if (intersects.length) {
            if (this.clicked === intersects[0].object) return;

            // @ts-ignore
            intersects[ 0 ].object.material.color.set( 0x880000 );

            this.hovered = intersects[0].object;
        } else {
            if (this.clicked === this.hovered) return;

            // @ts-ignore
            this.hovered.material.color.set(0xff0000)

            this.hovered = undefined;
        }
    }

    handleClick() {
        console.log("onclick")

        if (this.clicked) {
            // @ts-ignore
            this.clicked.material.color.set(0xff0000)

            this.clicked = undefined;
        }

        this.raycaster.setFromCamera(this.pointer, this.camera);

        const intersects = this.raycaster.intersectObjects(this.sphereMeshes);

        if (intersects.length) {
            this.controls.show();
            this.controls.set(intersects[0].object.position);

            // @ts-ignore
            intersects[ 0 ].object.material.color.set( 0x00ff00 );

            this.clicked = intersects[0].object;

            // @ts-ignore
            this.clickedIndex = this.sphereMeshes.findIndex((sphereMesh) => {
                return sphereMesh.id === this.clicked.id;
            })
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        // setTimeout(() => this.animate(), 1000);

        this.resizeCanvasToDisplaySize();

        if (this.clicked) {
            this.track[this.frame][this.clickedIndex] = this.controls.position.toArray();
        }

        if (this.track && this.skeletonHelper) {
            // this.frame = (this.frame + 1) % this.track.length;

            this.skeletonHelper.bones.forEach((bone, index) => {
                bone.position.fromArray(this.track[this.frame][index])
            });
        }

        if (this.sphereMeshes) this.sphereMeshes.forEach((mesh, index) => {
            const bone = this.skeletonHelper.bones[index];
            setSphereMesh(mesh, bone);
        })

        if (this.cylinderMeshes) this.cylinderMeshes.forEach((mesh, index) => {
            const bone = this.skeletonHelper.bones[index + 1];
            setCylinderMesh(mesh, bone);
        })

        this.handleIntersection();

        this.renderer.render(this.scene, this.camera);
    }

    constructSkeleton(motionSequence: number[][][]): [Skeleton, number[][][]] {
        const bones: Bone[] = [];

        const pos = motionSequence[0];
        const x = pos.map((p: number[]) => [
            p[0] - pos[0][0],
            p[1] - pos[0][1],
            p[2] - pos[0][2]
        ]);

        // const x = pos;

        parents.forEach((parent, index) => {
            const bone = new Bone();
            bone.name = `${names[index]}`;
            bone.position.fromArray(x[index]);
            bones.push(bone);

            if (parent >= 0)
                bones[parent].add(bone);
        });

        const skeleton = new Skeleton(bones);

        const tracks: number[][][] = [...Array(motionSequence.length)].map(() => []);

        motionSequence.forEach((positions: number[][], frameNo) => {
            positions.forEach((position, boneNo) => {
                tracks[frameNo].push([
                    position[0] - positions[0][0],
                    position[1] - positions[0][1],
                    position[2] - positions[0][2]
                ]);
                // tracks[index].push(position)
            })
        })

        return [skeleton, tracks];
    }
}

const setSphereMesh = (mesh: Mesh, bone: Bone) => {
    mesh.position.fromArray(bone.position.toArray()); // .add(bone.parent.position).divide(new Vector3(2, 2, 2))

    // console.log(bone.position.clone().sub(bone.parent.position).normalize())
}

const setCylinderMesh = (mesh: Mesh, bone: Bone) => {
    const parent = bone.parent.position.clone();
    const self = bone.position.clone();
    const d = self.distanceTo(parent);

    mesh.geometry.dispose();

    mesh.geometry = new CylinderGeometry(1.5, 1.5, d);
    
    mesh.position.fromArray(
        parent.clone().add(self.sub(parent).divide(new Vector3(2, 2, 2))).toArray()
    );

    mesh.lookAt(bone.position);
    mesh.rotateX(1.57);
}

const parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20];
// const permutation = [0, 9, 10, 11, 12, 13, 18, 19, 20, 21, 14, 15, 16, 17, 5, 6, 7, 8, 1, 2, 3, 4];
const permutation = [0, 14, 15, 16, 17, 18, 19, 20, 21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
const names = [
    "ModelHips",
    "ModelLeftUpLeg",
    "ModelLeftLeg",
    "ModelLeftFoot",
    "ModelLeftToe",
    "ModelRightUpLeg",
    "ModelRightLeg",
    "ModelRightFoot",
    "ModelRightToe",
    "ModelSpine",
    "ModelSpine1",
    "ModelSpine2",
    "ModelNeck",
    "ModelHead",
    "ModelLeftShoulder",
    "ModelLeftArm",
    "ModelLeftForeArm",
    "ModelLeftHand",
    "ModelRightShoulder",
    "ModelRightArm",
    "ModelRightForeArm",
    "ModelRightHand",
]