import { Clock } from 'three';
import RenderBVH from './renderBVH';

import './style.css';
import fetchJson from './helpers/fetchJson';

(async () => {
    const motionSequences = [
        await fetchJson('./static/animations/ground_truth.json') as number[][][], 
        await fetchJson('./static/animations/output.json') as number[][][]
    ];

    const container = document.createElement('div')
    container.classList.add('container')
    
    for (let i = 0; i < 2; i++) {
        const canvas = document.createElement('canvas')
        
        canvas.classList.add('canvas')
    
        container.appendChild(canvas);
        
        const clock = new Clock()
        
        new RenderBVH(canvas, motionSequences[i], clock, i)
    }
    
    document.body.appendChild(container)
})();