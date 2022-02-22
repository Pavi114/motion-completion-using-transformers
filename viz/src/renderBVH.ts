import { AmbientLight, AnimationClip, AnimationMixer, Bone, BoxGeometry, Clock, Color, GridHelper, Group, KeyframeTrack, Mesh, MeshStandardMaterial, PerspectiveCamera, Plane, PlaneGeometry, PlaneHelper, Scene, Skeleton, SkeletonHelper, Vector3, VectorKeyframeTrack, WebGLRenderer } from 'three';
import { BVH, BVHLoader } from 'three/examples/jsm/loaders/BVHLoader';

export default class RenderBVH {
    loader: BVHLoader;
    renderer: WebGLRenderer;
    skeletonHelper: SkeletonHelper;
    scene: Scene;
    mixer: AnimationMixer;
    camera: PerspectiveCamera;
    clock: Clock;
    id: number;

    constructor(canvas: HTMLCanvasElement, motionSequence: number[][][], clock: Clock, id: number) {
        this.clock = clock;
        this.id = id;

        this.init(canvas);
        this.animate();

        // this.loader.load(bvhFile, (result: BVH) => {
        //     console.log(result);

        //     this.skeletonHelper = new SkeletonHelper(result.skeleton.bones[0]);

        //     // @ts-ignore
        //     this.skeletonHelper.skeleton = result.skeleton

        //     this.scene.add(this.skeletonHelper);

        //     const boneContainer = new Group();
        //     boneContainer.add(result.skeleton.bones[0]);
        //     this.scene.add(boneContainer);

        //     this.mixer = new AnimationMixer(this.skeletonHelper);
            
        //     const animationClip = new AnimationClip(
        //         'animation',
        //         -1,
        //         result.clip.tracks.map(track => {
        //             const values = []
        //             for (let i = 0; i < track.values.length; i += 3) {
        //                 values.push(
        //                     track.values[i],
        //                     track.values[i + 1],
        //                     track.values[i + 2]
        //                 )
        //             }

        //             const times = [...Array(track.times.length)].map((_, i) => i / 30);

        //             return new VectorKeyframeTrack(
        //                 track.name,
        //                 times,
        //                 values
        //             )
        //         })
        //     )

        //     console.log(animationClip)

        //     console.log(result.clip)

        //     this.mixer.clipAction(animationClip).play();
        // })

        const [skeleton, animationClip] = this.constructSkeleton(motionSequence);

        this.skeletonHelper = new SkeletonHelper(skeleton.bones[0]);
        // @ts-ignore
        this.skeletonHelper.skeleton = skeleton

        this.scene.add(this.skeletonHelper);

        const boneContainer = new Group();
        boneContainer.add(skeleton.bones[0]);
        this.scene.add(boneContainer);

        this.mixer = new AnimationMixer(this.skeletonHelper);
        this.mixer.clipAction(animationClip).play();
    }

    init(canvas: HTMLCanvasElement) {
        this.camera = new PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 10000);
        this.camera.position.set(300, 300, 300);
        this.camera.lookAt(0, 0, 0);

        this.scene = new Scene();
        this.scene.background = new Color(0xeeeeee);

        this.scene.add(new AmbientLight());

        const ground = new Mesh(
            new PlaneGeometry(300, 300), 
            new MeshStandardMaterial({ color: 0x333333 })
        );
        ground.rotation.set(-Math.PI / 2, 0, 0);
        ground.position.set(0, -200, 0);
        this.scene.add(ground);


        this.renderer = new WebGLRenderer({ antialias: true, canvas: canvas });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);

        this.loader = new BVHLoader();
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
            bone.name = `bone${index}`;
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
                `.bones[bone${index}].position`,
                times,
                vector3Sequence
            )
            
            return vectorTrack
        })

        const animationClip = new AnimationClip('animation', -1, keyframeTracks);

        return [skeleton, animationClip];
    }
}

const parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20];