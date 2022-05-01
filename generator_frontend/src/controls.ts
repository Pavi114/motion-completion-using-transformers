import { Vector3 } from "three";

export default class Controls {
    position: Vector3;
    container: HTMLDivElement;
    sliders: HTMLInputElement[];

    constructor() {
        this.position = new Vector3(0, 0, 0);

        this.initHTML();
        this.initControls();
    }

    initHTML() {
        this.container = document.createElement('div');
        this.container.className = 'controls';

        const axes = ['x', 'y', 'z'];

        this.sliders = [];

        for (let i = 0; i < 3; i++) {
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.min = '-100';
            slider.max = '100';
            slider.id = `slider-${axes[i]}`
            this.sliders.push(slider);

            const label = document.createElement('label');
            label.innerText = axes[i];
            label.setAttribute('for', `slider-${axes[i]}`);

            this.container.appendChild(label);
            this.container.appendChild(slider);
        }

        document.body.appendChild(this.container);
    }

    initControls() {
        this.container.addEventListener('click', (event: MouseEvent) => {
            event.stopPropagation();
        })

        this.sliders.forEach((slider, index) => {
            slider.addEventListener('input', (event: InputEvent) => {
                // @ts-ignore
                this.position.setComponent(index, parseFloat(event.target.value));
            })
        })
    }

    show() {
        this.container.classList.remove('hidden');
    }

    hide() {
        this.container.classList.add('hidden');
    }

    set(position: Vector3) {
        this.position = position;

        this.sliders[0].value = position.x.toString()
        this.sliders[1].value = position.y.toString()
        this.sliders[2].value = position.z.toString()
    }
}