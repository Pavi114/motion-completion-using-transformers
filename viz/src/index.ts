import { Clock } from 'three';
import RenderBVH from './renderBVH';

import './style.css';
import fetchJson from './helpers/fetchJson';

(async () => {
    const animation_id = prompt("Enter animation id: ", "1");
    // const animation_id = 1;

    const motionSequences = [
        await fetchJson(`./static/animations/${animation_id}/ground_truth.json`) as number[][][], 
        await fetchJson(`./static/animations/${animation_id}/input.json`) as number[][][],
        await fetchJson(`./static/animations/${animation_id}/output.json`) as number[][][],
        await fetchJson(`./static/animations/${animation_id}/output_smoothened.json`) as number[][][]
    ];

    const container = document.createElement('div');
    container.classList.add('container');
    
    motionSequences.forEach((motionSequence, index) => {
        const canvas = document.createElement('canvas');
        
        canvas.classList.add('canvas');
    
        container.appendChild(canvas);
        
        const clock = new Clock();
        
        new RenderBVH(canvas, motionSequence, clock, index);
    });
    
    document.body.appendChild(container);
})();