import { AmbientLight, AnimationClip, AnimationMixer, Bone, BoxGeometry, Clock, Color, CylinderGeometry, GridHelper, Group, KeyframeTrack, LineBasicMaterial, Matrix3, Matrix4, Mesh, MeshStandardMaterial, PerspectiveCamera, Plane, PlaneGeometry, PlaneHelper, Scene, Skeleton, SkeletonHelper, SkinnedMesh, Sphere, SphereGeometry, Vector3, VectorKeyframeTrack, WebGLRenderer } from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader';

export default class RenderBVH {
    renderer: WebGLRenderer;
    skeletonHelper: SkeletonHelper;
    scene: Scene;
    mixer: AnimationMixer;
    camera: PerspectiveCamera;
    clock: Clock;
    id: number;
    sphereMeshes: Mesh[];
    cylinderMeshes: Mesh[];

    constructor(canvas: HTMLCanvasElement, motionSequence: number[][][], clock: Clock, id: number) {
        this.clock = clock;
        this.id = id;

        this.init(canvas);
        this.animate();

        const [skeleton, animationClip] = this.constructSkeleton(motionSequence);

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
        
        const sphereMaterial = new MeshStandardMaterial();
        sphereMaterial.color.setRGB(255, 0, 0);

        const cylinderMaterial = new MeshStandardMaterial();
        cylinderMaterial.color.setRGB(0, 0, 255);

        skeleton.bones.forEach((bone, index) => {
            const sphereGeometry = new SphereGeometry(3.2);

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
        
        this.mixer = new AnimationMixer(this.skeletonHelper);
        this.mixer.clipAction(animationClip).play();

        // const loader = new FBXLoader();
        // loader.load('./static/rp_eric_rigged_001_u3d.fbx', model => {
        //     model.traverse(child => {
        //         if (child instanceof SkinnedMesh) {
        //             // child.skeleton.bones = child.skeleton.bones.map((_, index) => child.skeleton.bones[permutation[index]]);

        //             console.log("FBX Bones | BVH Bones")
        //             child.skeleton.bones.forEach((bone, index) => console.log(bone.name, bone.parent.name, "|", skeleton.bones[index].name, skeleton.bones[index].parent.name))

        //             // console.log("Original BVH Bones")
        //             // skeleton.bones.forEach(bone => console.log(bone.name, bone.parent.name))

        //             // skeleton.pose()

        //             // child.skeleton.bones = skeleton.bones.map(
        //             //     (_, index) => {
        //             //         // console.log(skeleton.bones, index, skeleton.bones[index])
        //             //         return child.skeleton.getBoneByName(skeleton.bones[index].name)
        //             //     }
        //             // );

        //             // console.log(child.skeleton.bones[0])
        //             // console.log(skeleton.bones[0])

        //             // child.skeleton.bones = child.skeleton.bones.map((_, index) => skeleton.bones[permutation[index]]);

        //             // console.log("Permuted FBX Bones")
        //             // child.skeleton.bones.forEach(bone => console.log(bone.name, bone.parent.name))

        //             // skeleton.pose();
        //             child.bind(skeleton);
        //             // child.add(skeleton.bones[0]);

        //             console.log(child)

        //             console.log('\n\n\n')

        //             console.log("FBX Bones | BVH Bones")
        //             child.skeleton.bones.forEach((bone, index) => console.log(bone.name, bone.parent.name, "|", skeleton.bones[index].name, skeleton.bones[index].parent.name))

        //         }
        //     });

        //     // this.mixer = new AnimationMixer(model);

        //     this.scene.add(model);
        // })
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

    animate() {
        requestAnimationFrame(() => this.animate());

        this.resizeCanvasToDisplaySize();

        const delta = this.clock.getDelta();

        if (this.mixer) this.mixer.update(delta);

        if (this.sphereMeshes) this.sphereMeshes.forEach((mesh, index) => {
            const bone = this.skeletonHelper.bones[index];
            setSphereMesh(mesh, bone);
        })

        if (this.cylinderMeshes) this.cylinderMeshes.forEach((mesh, index) => {
            const bone = this.skeletonHelper.bones[index + 1];
            setCylinderMesh(mesh, bone);
        })

        this.renderer.render(this.scene, this.camera);
    }

    constructSkeleton(motionSequence: number[][][]): [Skeleton, AnimationClip] {
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

        const tracks: number[][][] = [...Array(22)].map(() => []);

        motionSequence.forEach((positions: number[][]) => {
            positions.forEach((position, index) => {
                tracks[index].push([
                    position[0] - positions[0][0],
                    position[1] - positions[0][1],
                    position[2] - positions[0][2]
                ]);
                // tracks[index].push(position)
            })
        })

        const times = [...Array(tracks[0].length)].map((_, i) => i * 0.033);

        const keyframeTracks = tracks.map((jointPositions, index) => {
            const vector3Sequence = jointPositions.flatMap(
                p => p
            )

            const vectorTrack = new VectorKeyframeTrack(
                `.bones[${names[index]}].position`,
                times,
                vector3Sequence
            )

            return vectorTrack
        })

        const animationClip = new AnimationClip('animation', -1, keyframeTracks);

        return [skeleton, animationClip];
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