import { Clock } from 'three';
import MotionEditor from './motionEditor';

import './style.css';
import fetchJson from './helpers/fetchJson';
import Controls from './controls';
import generateMotionSequence from './generate';
import RenderBVH from './renderBVH';

(async () => {
    // const animation_id = prompt("Enter animation id: ", "1");
    const animation_id = 1;

    const motionSequence = await fetchJson(`./static/animations/${animation_id}/ground_truth.json`) as number[][][];

    const container = document.createElement('div');
    container.classList.add('container');

    const controls = new Controls();
    
    const canvas = document.createElement('canvas');
    
    canvas.classList.add('canvas');

    container.appendChild(canvas);
    
    const clock = new Clock();
    
    const motionEditor = new MotionEditor(canvas, motionSequence, controls);
    
    document.body.appendChild(container);

    const generateButton = document.createElement('button');
    generateButton.className = 'generate';
    generateButton.innerText = 'Generate';

    document.body.appendChild(generateButton);

    generateButton.addEventListener('click', async () => {
        const res = await generateMotionSequence(motionEditor.track);

        const motionSequences = [
            res['z_x'],
            res['in_x']
        ];

        document.body.removeChild(container);
        document.body.removeChild(generateButton);

        motionEditor.controls.hide();

        const ncontainer = document.createElement('div');
        ncontainer.classList.add('container');
        
        motionSequences.forEach((motionSequence, index) => {
            const canvas = document.createElement('canvas');
            
            canvas.classList.add('canvas');
        
            ncontainer.appendChild(canvas);
            
            const clock = new Clock();
            
            new RenderBVH(canvas, motionSequence, clock, index);
        });
        
        document.body.appendChild(ncontainer);
    });
})();